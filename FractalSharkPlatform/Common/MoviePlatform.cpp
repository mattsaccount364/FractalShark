#include "MoviePlatform.h"
#include "Environment.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <exception>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace Environment {
namespace {
#ifdef _WIN32
class NativeFile {
public:
    HANDLE Value = INVALID_HANDLE_VALUE;
    ~NativeFile() { Close(); }
    void
    Close()
    {
        if (Value != INVALID_HANDLE_VALUE && Value != nullptr) {
            CloseHandle(Value);
            Value = INVALID_HANDLE_VALUE;
        }
    }
};
[[noreturn]] void
SystemFailure(const char *operation)
{
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
}
void
MakePipe(NativeFile &reader, NativeFile &writer)
{
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    if (!CreatePipe(&reader.Value, &writer.Value, &security, 0)) {
        SystemFailure("create movie process pipe");
    }
}
// Follow CommandLineToArgvW/CRT backslash rules, including trailing slashes.
std::wstring
QuoteArgument(const std::wstring &argument)
{
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++slashes;
        } else {
            result.append(slashes * (character == L'\"' ? 2 : 1), L'\\');
            slashes = 0;
            if (character == L'\"') {
                result += L'\\';
            }
            result += character;
        }
    }
    result.append(slashes * 2, L'\\');
    return result + L'\"';
}
#else
class NativeFile {
public:
    int Value = -1;
    ~NativeFile() { Close(); }
    void
    Close()
    {
        if (Value >= 0) {
            close(Value);
            Value = -1;
        }
    }
};
[[noreturn]] void
SystemFailure(const char *operation)
{
    throw std::system_error(errno, std::generic_category(), operation);
}
void
MakePipe(NativeFile &reader, NativeFile &writer)
{
    int handles[2];
    if (pipe2(handles, O_CLOEXEC) != 0) {
        SystemFailure("create movie process pipe");
    }
    reader.Value = handles[0];
    writer.Value = handles[1];
}
// An encoder exiting early must produce EPIPE, not kill the renderer. Block
// SIGPIPE only on the writing thread and consume only a signal we generated.
class ScopedPipeSignal {
public:
    ScopedPipeSignal()
    {
        sigemptyset(&m_Set);
        sigaddset(&m_Set, SIGPIPE);
        pthread_sigmask(SIG_BLOCK, &m_Set, &m_Previous);
        sigset_t pending;
        sigpending(&pending);
        m_WasPending = sigismember(&pending, SIGPIPE) != 0;
    }
    ~ScopedPipeSignal()
    {
        if (!m_WasPending) {
            const timespec immediate{0, 0};
            sigtimedwait(&m_Set, nullptr, &immediate);
        }
        pthread_sigmask(SIG_SETMASK, &m_Previous, nullptr);
    }

private:
    sigset_t m_Set;
    sigset_t m_Previous;
    bool m_WasPending;
};
#endif

void
ReadPipe(NativeFile &pipe, const std::function<void(std::span<const unsigned char>)> &consume)
{
    std::array<unsigned char, 65536> buffer;
    for (;;) {
#ifdef _WIN32
        DWORD bytes = 0;
        if (!ReadFile(pipe.Value, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes, nullptr)) {
            if (GetLastError() == ERROR_BROKEN_PIPE) {
                return;
            }
            SystemFailure("read movie process pipe");
        }
#else
        const auto bytes = read(pipe.Value, buffer.data(), buffer.size());
        if (bytes < 0) {
            if (errno == EINTR) {
                continue;
            }
            SystemFailure("read movie process pipe");
        }
#endif
        if (bytes == 0) {
            return;
        }
        consume(std::span<const unsigned char>(buffer.data(), static_cast<size_t>(bytes)));
    }
}
} // namespace

struct MovieJobLock::Impl {
    NativeFile File;
};

MovieJobLock::MovieJobLock(const std::filesystem::path &path) : m_Impl(std::make_unique<Impl>())
{
#ifdef _WIN32
    m_Impl->File.Value = CreateFileW(path.c_str(),
                                     GENERIC_READ | GENERIC_WRITE,
                                     0,
                                     nullptr,
                                     OPEN_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL,
                                     nullptr);
    if (m_Impl->File.Value == INVALID_HANDLE_VALUE) {
        SystemFailure("lock movie job (another process may own it)");
    }
#else
    m_Impl->File.Value = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (m_Impl->File.Value < 0 || flock(m_Impl->File.Value, LOCK_EX | LOCK_NB) != 0) {
        SystemFailure("lock movie job (another process may own it)");
    }
#endif
}
MovieJobLock::~MovieJobLock() = default;

void
FlushMovieFile(const std::filesystem::path &path)
{
    NativeFile file;
#ifdef _WIN32
    file.Value = CreateFileW(path.c_str(),
                             GENERIC_WRITE,
                             FILE_SHARE_READ,
                             nullptr,
                             OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    if (file.Value == INVALID_HANDLE_VALUE || !FlushFileBuffers(file.Value)) {
        SystemFailure("flush movie output");
    }
#else
    file.Value = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (file.Value < 0 || fsync(file.Value) != 0) {
        SystemFailure("flush movie output");
    }
    NativeFile directory;
    directory.Value = open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory.Value < 0 || fsync(directory.Value) != 0) {
        SystemFailure("flush movie frame directory");
    }
#endif
}

void
PublishMovieFile(const std::filesystem::path &temporary,
                 const std::filesystem::path &target,
                 bool replace)
{
    FlushMovieFile(temporary);
#ifdef _WIN32
    const DWORD flags = MOVEFILE_WRITE_THROUGH | (replace ? MOVEFILE_REPLACE_EXISTING : 0);
    if (!MoveFileExW(temporary.c_str(), target.c_str(), flags)) {
        SystemFailure("publish movie output");
    }
#else
    if (replace) {
        if (rename(temporary.c_str(), target.c_str()) != 0) {
            SystemFailure("publish movie output");
        }
    } else {
        if (link(temporary.c_str(), target.c_str()) != 0) {
            SystemFailure("publish movie output");
        }
        if (unlink(temporary.c_str()) != 0) {
            SystemFailure("remove published movie temporary");
        }
    }
    NativeFile directory;
    directory.Value = open(target.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory.Value < 0 || fsync(directory.Value) != 0) {
        SystemFailure("flush movie output directory");
    }
#endif
}

MovieProcessResult
RunMovieProcess(
    const std::filesystem::path &executable,
    const std::vector<std::string> &arguments,
    const std::function<void(const std::function<void(std::span<const unsigned char>)> &)> &input,
    const std::function<void(std::span<const unsigned char>)> &output,
    const std::function<void(std::string_view)> &errorOutput)
{
    NativeFile inputRead, inputWrite, outputRead, outputWrite, errorRead, errorWrite;
    MakePipe(inputRead, inputWrite);
    MakePipe(outputRead, outputWrite);
    MakePipe(errorRead, errorWrite);
#ifdef _WIN32
    if (!SetHandleInformation(inputWrite.Value, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(outputRead.Value, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(errorRead.Value, HANDLE_FLAG_INHERIT, 0)) {
        SystemFailure("protect movie process pipes");
    }
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = inputRead.Value;
    startup.StartupInfo.hStdOutput = outputWrite.Value;
    startup.StartupInfo.hStdError = errorWrite.Value;
    SIZE_T attributeBytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeBytes);
    std::vector<unsigned char> attributes(attributeBytes);
    startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &attributeBytes)) {
        SystemFailure("initialize movie process handles");
    }
    struct AttributeCleanup {
        LPPROC_THREAD_ATTRIBUTE_LIST Value;
        ~AttributeCleanup() { DeleteProcThreadAttributeList(Value); }
    } attributeCleanup{startup.lpAttributeList};
    HANDLE inherited[] = {inputRead.Value, outputWrite.Value, errorWrite.Value};
    if (!UpdateProcThreadAttribute(startup.lpAttributeList,
                                   0,
                                   PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   inherited,
                                   sizeof(inherited),
                                   nullptr,
                                   nullptr)) {
        SystemFailure("set movie process handles");
    }
    std::wstring command = QuoteArgument(executable.wstring());
    for (const auto &argument : arguments) {
        command += L" " + QuoteArgument(Environment::Utf8ToWide(argument));
    }
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(),
                        command.data(),
                        nullptr,
                        nullptr,
                        TRUE,
                        CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT,
                        nullptr,
                        nullptr,
                        &startup.StartupInfo,
                        &process)) {
        SystemFailure("start movie encoder");
    }
    NativeFile processHandle, threadHandle;
    processHandle.Value = process.hProcess;
    threadHandle.Value = process.hThread;
    const auto terminate = [&] { TerminateProcess(processHandle.Value, 1); };
#else
    posix_spawn_file_actions_t actions;
    int result = posix_spawn_file_actions_init(&actions);
    if (result != 0) {
        throw std::system_error(result, std::generic_category(), "initialize movie process");
    }
    struct ActionsCleanup {
        posix_spawn_file_actions_t *Value;
        ~ActionsCleanup() { posix_spawn_file_actions_destroy(Value); }
    } actionsCleanup{&actions};
    for (const auto pair : {std::pair{inputRead.Value, STDIN_FILENO},
                            std::pair{outputWrite.Value, STDOUT_FILENO},
                            std::pair{errorWrite.Value, STDERR_FILENO}}) {
        result = posix_spawn_file_actions_adddup2(&actions, pair.first, pair.second);
        if (result != 0) {
            throw std::system_error(result, std::generic_category(), "set movie process pipe");
        }
    }
    std::vector<std::string> storage{executable.string()};
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char *> argv;
    for (auto &argument : storage) {
        argv.push_back(argument.data());
    }
    argv.push_back(nullptr);
    pid_t process;
    result = posix_spawn(&process, executable.c_str(), &actions, nullptr, argv.data(), environ);
    if (result != 0) {
        throw std::system_error(result, std::generic_category(), "start movie encoder");
    }
    const auto terminate = [&] { kill(process, SIGKILL); };
    ScopedPipeSignal pipeSignal;
#endif
    inputRead.Close();
    outputWrite.Close();
    errorWrite.Close();
    MovieProcessResult response{-1, {}};
    std::exception_ptr outputError, errorError, inputError;
    std::jthread stdoutThread, stderrThread;
    // Declared after the thread owners: on unwind kill before joining readers,
    // including failure to construct the second thread. Reap only after their
    // callbacks finish, so a late callback cannot signal a recycled Unix PID.
    struct ChildCleanup {
        std::function<void()> Terminate;
        bool Finished = false;
        ~ChildCleanup()
        {
            if (!Finished) {
                Terminate();
            }
        }
    } childCleanup{[&] {
        terminate();
        if (stdoutThread.joinable()) {
            stdoutThread.join();
        }
        if (stderrThread.joinable()) {
            stderrThread.join();
        }
#ifdef _WIN32
        WaitForSingleObject(processHandle.Value, INFINITE);
#else
        int ignored;
        while (waitpid(process, &ignored, 0) < 0 && errno == EINTR) {
        }
#endif
    }};
    stdoutThread = std::jthread([&] {
        try {
            ReadPipe(outputRead, output);
        } catch (...) {
            outputError = std::current_exception();
            terminate();
        }
    });
    stderrThread = std::jthread([&] {
        try {
            ReadPipe(errorRead, [&](std::span<const unsigned char> bytes) {
                constexpr size_t ErrorTailBytes = 16384;
                response.ErrorOutput.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
                if (response.ErrorOutput.size() > ErrorTailBytes) {
                    response.ErrorOutput.erase(0, response.ErrorOutput.size() - ErrorTailBytes);
                }
                if (errorOutput) {
                    errorOutput(
                        std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
                }
            });
        } catch (...) {
            errorError = std::current_exception();
            terminate();
        }
    });
    try {
        input([&](std::span<const unsigned char> bytes) {
            while (!bytes.empty()) {
#ifdef _WIN32
                DWORD written = 0;
                const auto count = static_cast<DWORD>(std::min<size_t>(bytes.size(), 65536));
                if (!WriteFile(inputWrite.Value, bytes.data(), count, &written, nullptr)) {
                    SystemFailure("write movie process pipe");
                }
#else
                const auto written = write(inputWrite.Value, bytes.data(), bytes.size());
                if (written < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    SystemFailure("write movie process pipe");
                }
#endif
                if (written == 0) {
                    throw std::system_error(std::make_error_code(std::errc::broken_pipe));
                }
                bytes = bytes.subspan(static_cast<size_t>(written));
            }
        });
    } catch (...) {
        inputError = std::current_exception();
        terminate();
    }
    inputWrite.Close();
    stdoutThread.join();
    stderrThread.join();
#ifdef _WIN32
    WaitForSingleObject(processHandle.Value, INFINITE);
    DWORD code;
    if (!GetExitCodeProcess(processHandle.Value, &code)) {
        SystemFailure("wait for movie encoder");
    }
    response.ExitCode = static_cast<int>(code);
#else
    int status;
    while (waitpid(process, &status, 0) < 0) {
        if (errno != EINTR) {
            SystemFailure("wait for movie encoder");
        }
    }
    response.ExitCode = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
    childCleanup.Finished = true;
    if (outputError) {
        std::rethrow_exception(outputError);
    }
    if (errorError) {
        std::rethrow_exception(errorError);
    }
    if (inputError) {
        std::rethrow_exception(inputError);
    }
    return response;
}
} // namespace Environment
