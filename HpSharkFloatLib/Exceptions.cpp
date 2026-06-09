#include "Exceptions.h"

FractalSharkSeriousException::FractalSharkSeriousException(const char *msg)
{
#ifdef _MSC_VER
    if (msg == nullptr) {
        m_msg[0] = '\0';
    } else {
        strncpy_s(m_msg, msg, _TRUNCATE);
    }
#else
    if (msg == nullptr) {
        m_msg[0] = '\0';
    } else {
        std::strncpy(m_msg, msg, MsgBufSize - 1);
        m_msg[MsgBufSize - 1] = '\0';
    }
#endif
}
