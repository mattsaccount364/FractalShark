#include "CommandLineOptions.h"

#include <charconv>
#include <cstddef>
#include <string_view>
#include <type_traits>

namespace {

bool
ParseSignedInt(std::string_view text, int &value)
{
    if (text.empty()) {
        return false;
    }

    const char *first = text.data();
    const char *last = first + text.size();
    if (*first == '+') {
        ++first;
        if (first == last) {
            return false;
        }
    }

    const auto result = std::from_chars(first, last, value, 10);
    return result.ec == std::errc{} && result.ptr == last;
}

bool
ParseUnsignedInt(std::string_view text, uint32_t &value)
{
    if (text.empty()) {
        return false;
    }

    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

template <typename T>
bool
SetNumericOption(CommandLineOptionValue<T> &option,
                 std::string_view optionName,
                 std::string_view text,
                 std::string &error)
{
    if (option.m_Kind != CommandLineValueKind::Omitted) {
        error = "Duplicate command-line option --" + std::string(optionName);
        return false;
    }

    if (text == "auto") {
        option.m_Kind = CommandLineValueKind::Auto;
        return true;
    }

    T value{};
    const bool parsed = [&] {
        if constexpr (std::is_same_v<T, int>) {
            return ParseSignedInt(text, value);
        } else {
            return ParseUnsignedInt(text, value);
        }
    }();
    if (!parsed) {
        error = "Invalid value for --" + std::string(optionName) + ": " + std::string(text);
        return false;
    }

    option.m_Kind = CommandLineValueKind::Explicit;
    option.m_Value = value;
    return true;
}

bool
SetLimbOption(CommandLineOptionValue<uint32_t> &option,
              std::string_view optionName,
              std::string_view text,
              std::string &error)
{
    if (option.m_Kind != CommandLineValueKind::Omitted) {
        error = "Duplicate command-line option --" + std::string(optionName);
        return false;
    }

    if (text == "auto") {
        option.m_Kind = CommandLineValueKind::Auto;
        return true;
    }
    if (text == "production") {
        option.m_Kind = CommandLineValueKind::Production;
        return true;
    }

    uint32_t value = 0;
    if (!ParseUnsignedInt(text, value)) {
        error = "Invalid value for --" + std::string(optionName) + ": " + std::string(text);
        return false;
    }

    option.m_Kind = CommandLineValueKind::Explicit;
    option.m_Value = value;
    return true;
}

bool
SetVerboseOption(CommandLineOptionValue<int> &option,
                 std::string_view optionName,
                 std::string_view text,
                 std::string &error)
{
    if (text == "on") {
        return SetNumericOption(option, optionName, "1", error);
    }
    if (text == "off") {
        return SetNumericOption(option, optionName, "0", error);
    }
    return SetNumericOption(option, optionName, text, error);
}

bool
SetMpirThreadingOption(CommandLineOptionValue<int> &option,
                       std::string_view optionName,
                       std::string_view text,
                       std::string &error)
{
    if (text == "mt") {
        return SetNumericOption(option, optionName, "0", error);
    }
    if (text == "st") {
        return SetNumericOption(option, optionName, "1", error);
    }
    return SetNumericOption(option, optionName, text, error);
}

bool
SetOption(const std::string_view name,
          const std::string_view value,
          CommandLineOptions &options,
          std::string &error)
{
    if (name == "mode") {
        return SetNumericOption(options.m_Mode, name, value, error);
    }
    if (name == "verbose") {
        return SetVerboseOption(options.m_Verbose, name, value, error);
    }
    if (name == "cuda-iterations") {
        return SetNumericOption(options.m_CudaIterations, name, value, error);
    }
    if (name == "num-iters") {
        return SetNumericOption(options.m_NumIters, name, value, error);
    }
    if (name == "num-blocks") {
        return SetNumericOption(options.m_NumBlocks, name, value, error);
    }
    if (name == "num-threads") {
        return SetNumericOption(options.m_NumThreads, name, value, error);
    }
    if (name == "mpir-threading") {
        return SetMpirThreadingOption(options.m_MpirThreading, name, value, error);
    }
    if (name == "view") {
        return SetNumericOption(options.m_View, name, value, error);
    }
    if (name == "storage-limbs") {
        return SetLimbOption(options.m_StorageLimbs, name, value, error);
    }
    if (name == "effective-limbs") {
        return SetLimbOption(options.m_EffectiveLimbs, name, value, error);
    }

    error = "Unknown command-line option --" + std::string(name);
    return false;
}

} // namespace

CommandLineParseResult
ParseCommandLine(int argc, char **argv)
{
    CommandLineParseResult result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            result.m_ShowHelp = true;
            continue;
        }

        if (argument.size() < 3 || argument.substr(0, 2) != "--") {
            result.m_Error = "Unexpected command-line argument: " + std::string(argument);
            return result;
        }

        const std::string_view option = argument.substr(2);
        const size_t equals = option.find('=');
        std::string_view name = option;
        std::string_view value;
        if (equals != std::string_view::npos) {
            name = option.substr(0, equals);
            value = option.substr(equals + 1);
            if (value.empty()) {
                result.m_Error = "Missing value for --" + std::string(name);
                return result;
            }
        } else {
            if (i + 1 >= argc) {
                result.m_Error = "Missing value for --" + std::string(name);
                return result;
            }
            value = argv[++i];
        }

        if (name.empty() || !SetOption(name, value, result.m_Options, result.m_Error)) {
            return result;
        }
    }

    return result;
}

const char *
CommandLineUsage()
{
    return R"(HpSharkFloatTest options:
  --mode <1..13|auto>
  --verbose <0|1|on|off|auto>
  --cuda-iterations <integer|auto>
  --num-iters <integer|auto>
  --num-blocks <integer|auto>       (0 retains launch auto-selection)
  --num-threads <integer|auto>      (0 retains launch auto-selection)
  --mpir-threading <0|1|mt|st|auto>
  --view <1..34|auto>
  --storage-limbs <supported count|auto|production>
  --effective-limbs <integer|auto|production>
  --help

Both --name=value and --name value forms are accepted. Omitted options retain
the interactive prompts; auto selects the normal default without showing that
prompt. For full-reference views, auto uses the saved benchmark preset for
views 5, 30, and 32; production selects the coordinate-derived production
pair.
)";
}
