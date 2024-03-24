#pragma once

class JobObject {
public:
    JobObject();
    ~JobObject();

    JobObject &operator=(const JobObject &) = delete;
    JobObject(const JobObject &) = delete;

private:
    HANDLE hJob;
};