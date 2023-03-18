#ifndef FRACTALNETWORK_H
#define FRACTALNETWORK_H

#include <winsock.h>
#include <stdint.h>

const unsigned short PORTNUM = 41234;
const unsigned short PERM_PORTNUM = PORTNUM + 1;
const int BUFFER_SIZE = 8000;

class FractalNetwork
{
public:
  FractalNetwork ();
  ~FractalNetwork ();

  bool SetUpListener (const char *pcAddress, unsigned short portnum);
  bool AcceptConnection (sockaddr_in &sinRemote);

  u_long LookupAddress (const char *pcHost);
  SOCKET EstablishConnection (u_long nRemoteAddr, u_short nPort);
  bool CreateConnection (char *address, unsigned short portnum);

  int ReadData (char *buf, int n);
  int SendData (char *buf, int n);

  bool BufferedSendLong (uint32_t value);
  bool BufferedReadLong (uint32_t *value);

  bool BufferedSendShort (WORD value);
  bool BufferedReadShort (WORD *value);

  bool BufferedSendByte (BYTE value);
  bool BufferedReadByte (BYTE *value);

  bool BufferedReadEmpty (void);
  bool BufferedSendFlush (void);

  bool ShutdownConnection (void);

private:
  SOCKET m_ListenerSocket;
  SOCKET m_Socket;

  char m_SendBuffer[BUFFER_SIZE];
  int m_CurSendIndex;

  char m_ReadBuffer[BUFFER_SIZE];
  int m_CurReadIndex;
};

#endif