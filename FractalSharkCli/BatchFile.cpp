#include "stdafx.h"

#include "BatchFile.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string_view>

namespace FractalSharkCli {
namespace {

using Options = std::map<std::string, std::string>;

struct Section {
    std::string Name;
    Options Values;
    std::string Error;
};

std::string
Trim(std::string_view value)
{
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) {
        return {};
    }
    const size_t end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(begin, end - begin + 1));
}

bool
IsKnownKey(std::string_view key)
{
    constexpr std::string_view keys[] = {"width",
                                         "height",
                                         "render-algorithm",
                                         "out",
                                         "builtin-view",
                                         "locations",
                                         "location-index",
                                         "center-x",
                                         "center-y",
                                         "zoom",
                                         "iterations",
                                         "antialiasing",
                                         "perturbation-alg",
                                         "palette-map",
                                         "palette-depth",
                                         "console",
                                         "color",
                                         "quiet"};
    return std::find(std::begin(keys), std::end(keys), key) != std::end(keys);
}

bool
IsBooleanKey(std::string_view key)
{
    return key == "console" || key == "color" || key == "quiet";
}

int
SourceGroup(std::string_view key)
{
    if (key == "builtin-view") {
        return 1;
    }
    if (key == "locations") {
        return 2;
    }
    if (key == "center-x" || key == "center-y" || key == "zoom") {
        return 3;
    }
    return 0;
}

bool
IsValidName(std::string_view name)
{
    return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char character) {
        return std::isalnum(character) || character == '_' || character == '-';
    });
}

void
ReplaceName(std::string &value, std::string_view name)
{
    size_t pos = 0;
    while ((pos = value.find("{name}", pos)) != std::string::npos) {
        value.replace(pos, 6, name);
        pos += name.size();
    }
}

std::string
ResolvePath(const std::filesystem::path &directory, const std::string &value)
{
    return std::filesystem::absolute(directory / std::filesystem::path(value))
        .lexically_normal()
        .string();
}

} // namespace

bool
LoadBatchFile(const std::filesystem::path &path, BatchFile &batch, std::string &error)
{
    batch.Images.clear();
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open batch file: " + path.string();
        return false;
    }

    Section defaults;
    std::vector<Section> images;
    Section *current = nullptr;
    bool sawDefaults = false;
    std::string line;
    size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (lineNumber == 1 && line.starts_with("\xef\xbb\xbf")) {
            line.erase(0, 3);
        }
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed.front() == '#' || trimmed.front() == ';') {
            continue;
        }
        if (trimmed.front() == '[') {
            if (trimmed == "[defaults]") {
                if (sawDefaults || !images.empty()) {
                    error = "[defaults] must appear once before all images (line " +
                            std::to_string(lineNumber) + ")";
                    return false;
                }
                sawDefaults = true;
                current = &defaults;
            } else if (trimmed.starts_with("[image ") && trimmed.back() == ']') {
                images.push_back(
                    {Trim(std::string_view(trimmed).substr(7, trimmed.size() - 8)), {}, {}});
                current = &images.back();
                if (!IsValidName(current->Name)) {
                    current->Error = "image name must contain only letters, digits, '_' or '-'";
                }
            } else {
                error = "invalid batch section at line " + std::to_string(lineNumber);
                return false;
            }
            continue;
        }
        if (!current) {
            error = "batch option precedes a section at line " + std::to_string(lineNumber);
            return false;
        }
        const size_t separator = trimmed.find('=');
        if (separator == std::string::npos) {
            if (current == &defaults) {
                error = "expected key = value at line " + std::to_string(lineNumber);
                return false;
            }
            if (current->Error.empty()) {
                current->Error = "expected key = value at line " + std::to_string(lineNumber);
            }
            continue;
        }
        const std::string key = Trim(std::string_view(trimmed).substr(0, separator));
        const std::string value = Trim(std::string_view(trimmed).substr(separator + 1));
        if (!IsKnownKey(key) || current->Values.contains(key)) {
            const std::string reason =
                !IsKnownKey(key) ? "unknown batch option: " + key : "duplicate batch option: " + key;
            if (current == &defaults) {
                error = reason + " at line " + std::to_string(lineNumber);
                return false;
            }
            if (current->Error.empty()) {
                current->Error = reason + " at line " + std::to_string(lineNumber);
            }
            continue;
        }
        current->Values.emplace(key, value);
    }
    if (input.bad()) {
        error = "error reading batch file: " + path.string();
        return false;
    }
    if (images.empty()) {
        error = "batch file contains no [image name] sections";
        return false;
    }

    int defaultSource = 0;
    for (const auto &[key, value] : defaults.Values) {
        if (IsBooleanKey(key) && value != "true" && value != "false") {
            error = "invalid [defaults] value: " + key + " must be true or false";
            return false;
        }
        const int source = value.empty() ? 0 : SourceGroup(key);
        if (source != 0 && defaultSource != 0 && source != defaultSource) {
            error = "[defaults] has conflicting view sources";
            return false;
        }
        if (source != 0) {
            defaultSource = source;
        }
    }

    std::set<std::string> names;
    const auto directory = std::filesystem::absolute(path).parent_path();
    for (const Section &section : images) {
        BatchImage image;
        image.Name = section.Name;
        image.Error = section.Error;
        if (!names.insert(section.Name).second && image.Error.empty()) {
            image.Error = "duplicate image name: " + section.Name;
        }

        Options effective = defaults.Values;
        int imageSource = 0;
        for (const auto &[key, value] : section.Values) {
            const int source = value.empty() ? 0 : SourceGroup(key);
            if (source != 0 && imageSource != 0 && source != imageSource && image.Error.empty()) {
                image.Error = "image has conflicting view sources";
            }
            if (source != 0) {
                imageSource = source;
            }
        }
        if (imageSource != 0) {
            for (auto it = effective.begin(); it != effective.end();) {
                const int source = SourceGroup(it->first);
                if (source != 0 && source != imageSource) {
                    it = effective.erase(it);
                } else {
                    ++it;
                }
            }
            if (imageSource != 2) {
                effective.erase("location-index");
            }
        }
        for (const auto &[key, value] : section.Values) {
            if (value.empty()) {
                effective.erase(key);
            } else {
                effective[key] = value;
            }
        }

        for (auto &[key, value] : effective) {
            if (IsBooleanKey(key)) {
                if (value != "true" && value != "false" && image.Error.empty()) {
                    image.Error = key + " must be true or false";
                }
                continue;
            }
            try {
                if (key == "out") {
                    ReplaceName(value, image.Name);
                }
                if (key == "out" || key == "locations" || key == "palette-map") {
                    value = ResolvePath(directory, value);
                }
            } catch (const std::exception &exception) {
                if (image.Error.empty()) {
                    image.Error = key + ": " + exception.what();
                }
            }
        }
        if (effective.contains("out")) {
            image.Output = effective.at("out");
        }
        if (effective.contains("color") && effective.at("color") == "true" &&
            effective.contains("console") && effective.at("console") == "false" && image.Error.empty()) {
            image.Error = "color=true conflicts with console=false";
        }
        for (const auto &[key, value] : effective) {
            if (!IsBooleanKey(key) || value == "true") {
                image.Arguments.push_back("--" + key);
                if (!IsBooleanKey(key)) {
                    image.Arguments.push_back(value);
                }
            }
        }
        batch.Images.push_back(std::move(image));
    }
    return true;
}

} // namespace FractalSharkCli
