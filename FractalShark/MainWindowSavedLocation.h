#pragma once

struct MainWindow::SavedLocation {

    explicit SavedLocation(std::ifstream &infile)
    {
        // If anything fails, put the stream into a failed state and return
        // so the caller can stop reading.
        if (!infile.good()) {
            infile.setstate(std::ios::failbit);
            return;
        }

        HighPrecision minX, minY, maxX, maxY;

        // Read required fields
        infile >> width >> height;
        infile >> minX >> minY >> maxX >> maxY;
        infile >> num_iterations >> antialiasing;

        // If any of the above failed, bail early
        if (!infile.good()) {
            infile.setstate(std::ios::failbit);
            return;
        }

        // IMPORTANT: consume trailing whitespace before getline
        infile >> std::ws;
        std::getline(infile, description);

        // Construct after successful parse
        ptz = PointZoomBBConverter(minX, minY, maxX, maxY);
    }

    PointZoomBBConverter ptz{};
    size_t width = 0, height = 0;
    IterTypeFull num_iterations = 0;
    uint32_t antialiasing = 0;
    std::string description;
};

struct MainWindow::ImaginaSavedLocation {
    std::wstring Filename;
    ImaginaSettings Settings;
};
