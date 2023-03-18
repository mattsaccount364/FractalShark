#include "stdafx.h"
#include "FractalNetwork.h"
#include <stdio.h>

FractalNetwork::FractalNetwork ()
{ m_CurSendIndex = 0;
  m_CurReadIndex = 0;
  m_ListenerSocket = INVALID_SOCKET;
  m_Socket = INVALID_SOCKET;
}

FractalNetwork::~FractalNetwork ()
{
}

//// SetUpListener /////////////////////////////////////////////////////
// Sets up a listener on the given interface and port, returning the
// listening socket if successful; if not, returns INVALID_SOCKET.
bool FractalNetwork::SetUpListener (const char *pcAddress, unsigned short portnum)
{ u_long nInterfaceAddr = inet_addr (pcAddress);
  if (nInterfaceAddr != INADDR_NONE)
  { m_ListenerSocket = socket (AF_INET, SOCK_STREAM, 0);
    if (m_ListenerSocket != INVALID_SOCKET)
    { sockaddr_in sinInterface;
      sinInterface.sin_family = AF_INET;
      sinInterface.sin_addr.s_addr = nInterfaceAddr;
      sinInterface.sin_port = htons (portnum);
      if (bind (m_ListenerSocket, (sockaddr *) &sinInterface, sizeof (sockaddr_in)) != SOCKET_ERROR)
      { listen (m_ListenerSocket, 1);
        return true;
      }
    }
  }

  return false;
}

//// AcceptConnection //////////////////////////////////////////////////
// Waits for a connection on the given socket.  When one comes in, we
// return a socket for it.  If an error occurs, we return
// INVALID_SOCKET.
bool FractalNetwork::AcceptConnection (sockaddr_in &sinRemote)
{ int nAddrSize = sizeof (sinRemote);
  m_Socket = accept (m_ListenerSocket, (sockaddr *) &sinRemote, &nAddrSize);

  if (m_Socket == INVALID_SOCKET)
  { return false; }

  return true;
}

//// LookupAddress /////////////////////////////////////////////////////
// Given an address string, determine if it's a dotted-quad IP address
// or a domain address.  If the latter, ask DNS to resolve it.  In
// either case, return resolved IP address.  If we fail, we return
// INADDR_NONE.
u_long FractalNetwork::LookupAddress (const char* pcHost)
{ u_long nRemoteAddr = inet_addr (pcHost);
  if (nRemoteAddr == INADDR_NONE) {
    // pcHost isn't a dotted IP, so resolve it through DNS
    hostent* pHE = gethostbyname (pcHost);
    if (pHE == 0) {
      return INADDR_NONE;
    }
    nRemoteAddr = *((u_long *) pHE->h_addr_list[0]);
  }

  return nRemoteAddr;
}

//// EstablishConnection ///////////////////////////////////////////////
// Connects to a given address, on a given port, both of which must be
// in network byte order.  Returns newly-connected socket if we succeed,
// or INVALID_SOCKET if we fail.
SOCKET FractalNetwork::EstablishConnection (u_long nRemoteAddr, u_short nPort)
{ // Create a stream socket
  m_Socket = socket (AF_INET, SOCK_STREAM, 0);
  if (m_Socket != INVALID_SOCKET) {
    sockaddr_in sinRemote;
    sinRemote.sin_family = AF_INET;
    sinRemote.sin_addr.s_addr = nRemoteAddr;
    sinRemote.sin_port = nPort;

    if (connect (m_Socket, (sockaddr *) &sinRemote, sizeof (sockaddr_in)) == SOCKET_ERROR)
    { m_Socket = INVALID_SOCKET; }
  }

  return m_Socket;
}

bool FractalNetwork::CreateConnection (char *address, unsigned short portnum)
{ // Get a usable address from the string given.
  // e.g. convert 192.168.0.1 to an internal representation
  u_long nRemoteAddress = LookupAddress (address);
  if (nRemoteAddress == INADDR_NONE)
  { return false; }

  // Create the connection with the remote server
  m_Socket = EstablishConnection (nRemoteAddress, htons (portnum));
  if (m_Socket == INVALID_SOCKET)
  { return false; }

  return true;
}

int FractalNetwork::ReadData (char *buf, int n)
{ int bcount; /* counts bytes read */
  int br;     /* bytes read this pass */

  bcount = 0;
  br = 0;
  while (bcount < n) /* loop until full buffer */
  { if ((br = recv (m_Socket, buf, n - bcount, 0)) > 0)
    { bcount += br;  /* increment byte counter */
      buf += br;     /* move buffer ptr for next read */
    }
    else if (br < 0) /* signal an error to the caller */
    { return -1; }
    else
    { break; }
  }

  return bcount;
}

int FractalNetwork::SendData (char *buf, int n)
{ int bcount; /* counts bytes read */
  int br;     /* bytes read this pass */

  bcount = 0;
  br = 0;
  while (bcount < n) /* loop until all bytes sent */
  { if ((br = send (m_Socket, buf, n - bcount, 0)) > 0)
    { bcount += br;  /* increment byte counter */
      buf += br;     /* move buffer ptr for next read */
    }
    else if (br < 0) /* signal an error to the caller */
    { return -1; }
    else
    { break; }
  }

  return bcount;
}

bool FractalNetwork::BufferedSendLong (uint32_t value)
{ bool success = true;

  success &= BufferedSendByte (LOBYTE (LOWORD (value)));
  success &= BufferedSendByte (HIBYTE (LOWORD (value)));
  success &= BufferedSendByte (LOBYTE (HIWORD (value)));
  success &= BufferedSendByte (HIBYTE (HIWORD (value)));

  return success;
}

bool FractalNetwork::BufferedReadLong (uint32_t *value)
{ bool success = true;

  BYTE c1, c2, c3, c4;
  success &= BufferedReadByte (&c1);
  success &= BufferedReadByte (&c2);
  success &= BufferedReadByte (&c3);
  success &= BufferedReadByte (&c4);

  *value = MAKELONG (MAKEWORD (c1, c2), MAKEWORD (c3, c4));

  return success;
}

bool FractalNetwork::BufferedSendShort (WORD value)
{ bool success = true;

  success &= BufferedSendByte (LOBYTE (value));
  success &= BufferedSendByte (HIBYTE (value));

  return success;
}

bool FractalNetwork::BufferedReadShort (WORD *value)
{ bool success = true;

  BYTE c1, c2;
  success &= BufferedReadByte (&c1);
  success &= BufferedReadByte (&c2);

  *value = MAKEWORD (c1, c2);

  return success;
}

bool FractalNetwork::BufferedSendByte (BYTE value)
{ if (m_CurSendIndex + 1 <= BUFFER_SIZE)
  { m_SendBuffer[m_CurSendIndex] = (char) value;
    m_CurSendIndex++;
  }

  bool success = true;
  if (m_CurSendIndex == BUFFER_SIZE)
  { success = (SendData (m_SendBuffer, BUFFER_SIZE) < 0) ? false : true;
    m_CurSendIndex = 0;
  }

  if (success == false)
  { return false; }

  return true;
}

bool FractalNetwork::BufferedReadByte (BYTE *value)
{ bool success = true;
  if (m_CurReadIndex == 0 || m_CurReadIndex == BUFFER_SIZE)
  { success = (ReadData (m_ReadBuffer, BUFFER_SIZE) < 0) ? false : true;
    m_CurReadIndex = 0;
  }

  if (success == false)
  { return false; }

  *value = m_ReadBuffer[m_CurReadIndex];
  m_CurReadIndex++;

  return true;
}

bool FractalNetwork::BufferedReadEmpty (void)
{ m_CurReadIndex = 0;
  return true;
}

bool FractalNetwork::BufferedSendFlush (void)
{ bool success;
  success = (SendData (m_SendBuffer, BUFFER_SIZE) < 0) ? false : true;
  m_CurSendIndex = 0;
  return success;
}

//// ShutdownConnection ////////////////////////////////////////////////
// Gracefully shuts the connection sd down.  Returns true if we're
// successful, false otherwise.
bool FractalNetwork::ShutdownConnection (void)
{ // Disallow any further data sends.  This will tell the other side
  // that we want to go away now.  If we skip this step, we don't
  // shut the connection down nicely.
  if (shutdown (m_Socket, 0x01) == SOCKET_ERROR) {
    return false;
  }

  // Receive any extra data still sitting on the socket.  After all
  // data is received, this call will block until the remote host
  // acknowledges the TCP control packet sent by the shutdown above.
  // Then we'll get a 0 back from recv, signalling that the remote
  // host has closed its side of the connection.
  char acReadBuffer[1024];
  for (;;)
  { int nNewBytes = recv (m_Socket, acReadBuffer, 1024, 0);
    if (nNewBytes == SOCKET_ERROR) {
      return false;
    }
    else if (nNewBytes != 0) {
      printf ("FYI, received %d unexpected bytes during shutdown.\r\n", nNewBytes);
    }
    else {
      // Okay, we're done!
      break;
    }
  }

  // Close the socket.
  if (closesocket (m_Socket) == SOCKET_ERROR) {
    return false;
  }

  return true;
}