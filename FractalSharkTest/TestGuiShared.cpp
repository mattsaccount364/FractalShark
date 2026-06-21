#include "CommandCatalog.h"
#include "GuiHelp.h"
#include "SavedLocation.h"
#include "TestFramework.h"

#include <sstream>

TEST(GuiHelp_HotkeysContainsEveryCatalogEntry)
{
    const std::string help = FractalShark::BuildHotkeysHelpUtf8();
    for (const FractalShark::Command &command : FractalShark::kCommands) {
        const std::string hotkey = FractalShark::FormatHotKeyUtf8(command.hotkey);
        ASSERT_TRUE(help.find(hotkey + " - ") != std::string::npos);
    }
    ASSERT_TRUE(help.find("Direct controls") != std::string::npos);
    ASSERT_TRUE(FractalShark::BuildHotkeysHelpWide().find(L"\r\n") != std::wstring::npos);
}

TEST(GuiHelp_UsesSharedTopicContent)
{
    const auto views = FractalShark::GetGuiHelpContent(FractalShark::GuiHelpTopic::Views);
    const auto algorithms = FractalShark::GetGuiHelpContent(FractalShark::GuiHelpTopic::Algorithms);
    ASSERT_EQ(views.Title, "Views");
    ASSERT_EQ(algorithms.Title, "Algorithms");
    ASSERT_TRUE(views.Body.find("interesting locations") != std::string_view::npos);
    ASSERT_TRUE(algorithms.Body.find("choose AUTO") != std::string_view::npos);
}

TEST(SavedLocation_RoundTripsPortableFormat)
{
    FractalShark::SavedLocation original;
    original.Width = 1920;
    original.Height = 1080;
    original.MinX = HighPrecision("-2.125");
    original.MinY = HighPrecision("-1.25");
    original.MaxX = HighPrecision("0.875");
    original.MaxY = HighPrecision("1.25");
    original.NumIterations = 123456789;
    original.Antialiasing = 4;
    original.Description = "test location with spaces";

    const std::string serialized = FractalShark::SerializeSavedLocation(original);
    std::istringstream input(serialized + "\r\n");
    FractalShark::SavedLocation parsed;
    ASSERT_TRUE(FractalShark::ParseSavedLocation(input, parsed));
    ASSERT_EQ(parsed.Width, original.Width);
    ASSERT_EQ(parsed.Height, original.Height);
    ASSERT_EQ(parsed.MinX.str(), original.MinX.str());
    ASSERT_EQ(parsed.MinY.str(), original.MinY.str());
    ASSERT_EQ(parsed.MaxX.str(), original.MaxX.str());
    ASSERT_EQ(parsed.MaxY.str(), original.MaxY.str());
    ASSERT_EQ(parsed.NumIterations, original.NumIterations);
    ASSERT_EQ(parsed.Antialiasing, original.Antialiasing);
    ASSERT_EQ(parsed.Description, original.Description);
}

TEST(SavedLocation_RejectsTruncatedRecordAndHonorsLimit)
{
    std::istringstream truncated("1920 1080 -2 -1 1");
    FractalShark::SavedLocation location;
    ASSERT_FALSE(FractalShark::ParseSavedLocation(truncated, location));

    const std::string row = "10 20 -2 -1 1 2 100 1 label\n";
    std::istringstream multiple(row + row + row);
    const auto locations = FractalShark::ReadSavedLocations(multiple, 2);
    ASSERT_EQ(locations.size(), 2u);
}

TEST(SavedLocation_PreservesEmptyDescriptions)
{
    const std::string first = "10 20 -2 -1 1 2 100 1 \r\n";
    const std::string second = "30 40 -3 -2 2 3 200 2 named\r\n";
    std::istringstream input(first + second);
    const auto locations = FractalShark::ReadSavedLocations(input, 30);
    ASSERT_EQ(locations.size(), 2u);
    ASSERT_TRUE(locations[0].Description.empty());
    ASSERT_EQ(locations[1].Description, "named");
}
