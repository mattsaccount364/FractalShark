#include "stdafx.h"
#include <windows.h>

#include <winsock.h>
#include <stdio.h>
#include <conio.h>

#include "..\Fractal.h"
#include "..\FractalNetwork.h"
#include "..\FractalSetupData.h"

// Holds data that the user can customize, e.g. the IP
// address of the network interface to use.
FractalSetupData gSetupData;

FractalNetwork gExitNetwork;
FractalNetwork gMainNetwork;
FractalNetwork gSubNetwork;

HANDLE gHandleMainThread;
HANDLE gHandleSubThread;

// Holds the iterations to transmit
DWORD *gItersArray[4096];
BYTE *gProcessPixel[4096];

/* CheckKey - Thread to wait for a keystroke, then clear repeat flag. */
unsigned long WINAPI MainConnectionThread (void *);
unsigned long WINAPI SubConnectionThread (void *);

int ManageMainState (void);
bool BeginCalculations (void);

int main (int, char *[])
{ // Lower our priority a bit--this isn't mission critical :)
  SetThreadPriority (GetCurrentThread (), THREAD_PRIORITY_LOWEST);

  // Load up some user customization data.
  gSetupData.Load ();

  // Allocate the iterations array.
  int i, j;
  for (i = 0; i < 4096; i++)
  { gItersArray[i] = new DWORD[4096];
    for (j = 0; j < 4096; j++)
    { gItersArray[i][j] = 0; }
  }

  // Allocate the ProcessPixel array
  for (i = 0; i < 4096; i++)
  { gProcessPixel[i] = new BYTE[4096];
    for (j = 0; j < 4096; j++)
    { gProcessPixel[i][j] = 0; }
  }

  // Start listening for connections on the interface specified.
  WSADATA info;
  if (WSAStartup (MAKEWORD (1, 1), &info) != 0)
  { printf ("Cannot initialize WinSock!", "WSAStartup", MB_OK);
    return 0;
  }

  DWORD threadID;
  gHandleMainThread = (HANDLE) CreateThread (NULL, 0, MainConnectionThread, NULL, 0, &threadID);
  gHandleSubThread = (HANDLE) CreateThread (NULL, 0, SubConnectionThread, NULL, 0, &threadID);

  for (;;)
  { if (_kbhit () != 0)
    { getch ();
      break;
    }
  }

  printf ("Exiting...\r\n");

  char quittime[512];
  strcpy (quittime, "exit");

  gExitNetwork.CreateConnection ("127.0.0.1", PORTNUM);
  gExitNetwork.SendData (quittime, 512);
  gExitNetwork.ShutdownConnection ();

  WaitForSingleObject (gHandleSubThread, INFINITE);
  
  gExitNetwork.CreateConnection ("127.0.0.1", PERM_PORTNUM);
  gExitNetwork.SendData (quittime, 512);
  gExitNetwork.ShutdownConnection ();

  WaitForSingleObject (gHandleMainThread, INFINITE);

  printf ("Press a key...\r\n");
  getch ();

  WSACleanup ();

  // Clean up
  for (i = 0; i < 4096; i++)
  { delete [] gItersArray[i]; }
  for (i = 0; i < 4096; i++)
  { delete [] gProcessPixel[i]; }

  return 0;
}

unsigned long WINAPI MainConnectionThread (void *)
{ if (gMainNetwork.SetUpListener (gSetupData.m_ServerIP, PERM_PORTNUM) == false)
  { printf ("MainConnectionThread - SetUpListener failed");
    return 0;
  }

  int ret;
  sockaddr_in sinRemote;
  for (;;)
  { printf ("Awaiting primary connection...\r\n");

    if (gMainNetwork.AcceptConnection (sinRemote) == false)
    { printf ("Error waiting for new connection!\n");
      break;
    }

    printf ("Accepted!\r\n");

    do
    { ret = ManageMainState ();
    } while (ret == 1);

    gMainNetwork.ShutdownConnection ();

    if (ret == -1)
    { break; }
  }

  return 0;
}

// -1 = time to exit this program
// 0 = that client is done, we can wait for a new client now
// 1 = that client is not done, so we will keep running and await
//     more work to do.
int ManageMainState (void)
{ char buffer[512];
  gMainNetwork.ReadData (buffer, 512);

  printf ("ManageMainState - Data received:\r\n");
  printf (buffer);
  printf ("\r\n");

  if (strcmp (buffer, "exit") == 0)
  { return -1; }      // Time to exit this program
  else if (strcmp (buffer, "done") == 0)
  { return 0; }       // The client is done, we can wait for a new client
  else if (strcmp (buffer, "Initialize 1.1 step 1") == 0)
  { char ack[512];
    strcpy (ack, "Fractal Server 1.1");

    int ret = gMainNetwork.SendData (ack, 512);
    if (ret == -1)
    { printf ("Failure sending acknowledgement: %d\r\n", ret);
      return false;
    }

    printf ("Acknowledgement sent.  Awaiting initialization data...\r\n");
    return 1;
  }
  else if (strcmp (buffer, "Initialize 1.1 step 2") == 0)
  { int i, j;
    gMainNetwork.BufferedReadEmpty ();

    for (i = 0; i < 4096; i++)
    { for (j = 0; j < 4096; j++)
      { gMainNetwork.BufferedReadByte (&gProcessPixel[i][j]); }
    }

    return 1;
  }

  return 0;
}

unsigned long WINAPI SubConnectionThread (void *)
{ if (gSubNetwork.SetUpListener (gSetupData.m_ServerIP, PORTNUM) == false)
  { printf ("SubConnectionThread - SetUpListener failed");
    return 0;
  }

  // Wait for connections forever, until the client tells the server to shutdown
  sockaddr_in sinRemote;
  for (;;)
  { printf ("Awaiting secondary connection...\r\n");

    if (gSubNetwork.AcceptConnection (sinRemote) == false)
    { printf ("Error waiting for new connection!\n");
      break;
    }

    printf ("Accepted!\r\n");

    bool ret = BeginCalculations ();

    gSubNetwork.ShutdownConnection ();

    if (ret == false)
    { break; }
  }

  return 0;
}

bool BeginCalculations (void)
{ printf ("Awaiting instructions...\r\n");
  char buffer[512];
  gSubNetwork.ReadData (buffer, 512);

  int ScreenWidth, ScreenHeight;
  DWORD NumIters;
  double MinX, MaxX, MinY, MaxY, dx, dy;
  char ChangedItersOnly;

  printf ("Data received:\r\n");
  printf (buffer);
  printf ("\r\n");

  // Secondary connection should quit.
  if (strcmp (buffer, "exit") == 0)
  { return false; }

  // Anything else must be data for setting up a calculation.
  sscanf (buffer, "%d %d %d %lf %lf %lf %lf %c",
      &NumIters, &ScreenWidth, &ScreenHeight,
      &MinX, &MaxX, &MinY, &MaxY, &ChangedItersOnly);

  printf ("Received instructions.\r\n");
  printf ("Interpretation:\r\n");
  printf ("NumIters:     %d\r\n", NumIters);
  printf ("ScreenWidth:  %d\r\n", ScreenWidth);
  printf ("ScreenHeight: %d\r\n", ScreenHeight);
  printf ("MinX:         %.15f\r\n", MinX);
  printf ("MaxX:         %.15f\r\n", MaxX);
  printf ("MinY:         %.15f\r\n", MinY);
  printf ("MaxY:         %.15f\r\n", MaxY);
  printf ("ChangedItersOnly: %c\r\n", ChangedItersOnly);

  dx = (MaxX - MinX) / ScreenWidth;
  dy = -(MaxY - MinY) / ScreenHeight;

  // NumIters = maximum number of iterations to use when deciding whether
  //   a given point is within the fractal or not.
  // ScreenWidth / ScreenHeight = dimensions of the block we're looking at,
  //   e.g. 1024 x 768
  // MinX .. MaxY = The actual part of the fractal we're looking at.
  // dx / dy = The increment to use (could be calculated here but isn't)
  // ChangedItersOnly = A major speed optimization.  Allows just certain parts of the fractal
  //   to be recalculated when all that has changed is the number of iterations.

  // We don't want any custom settings loaded...
  // just avoid "network rendering", since we are the server!
  // We do network rendering for others, not ourselves.
  FractalSetupData setupData;
  setupData.Load (true);
  setupData.m_NetworkRender = false;
  Fractal fractal (&setupData, 0, 0);

  double x, y;
  int px, py;

  DWORD prevMaxIters = 0;
  if (ChangedItersOnly == 'y')
  { // Determine the maximum number of iterations used previously.
    for (py = 0; py < ScreenHeight; py++)
    { for (px = 0; px < ScreenWidth; px++)
      { if (gProcessPixel[px][py] == 's')
        { if (gItersArray[px][py] > prevMaxIters)
          { prevMaxIters = gItersArray[px][py]; }
        }
      }
    }
  }

  py = 0;
  y = MaxY;
  while (py < ScreenHeight)
  { px = 0;
    x = MinX;
    while (px < ScreenWidth)
    { if (gProcessPixel[px][py] == 's')
      { if ((ChangedItersOnly == 'n') ||
            (ChangedItersOnly == 'y' &&
             (gItersArray[px][py] == prevMaxIters ||
              gItersArray[px][py] >= NumIters)
            )
           )
        { gItersArray[px][py] = fractal.WithinFractal (x, y, NumIters); }
      }

      x += dx;
      px++;
    }

    printf ("%2.1f%%\b\b\b\b\b", (double) py / ScreenHeight * 100.0);

    y += dy;
    py++;
  }

  printf ("Calculations done.  Sending results...\r\n");

  for (py = 0; py < ScreenHeight; py++)
  { for (px = 0; px < ScreenWidth; px++)
    { if (gProcessPixel[px][py] != 's')
      { continue; }

      if (NumIters >= 65536)
      { if (gSubNetwork.BufferedSendLong (gItersArray[px][py]) == false)
        { printf ("Failure sending data...\r\n");
          return false;
        }
      }
      else
      { if (gSubNetwork.BufferedSendShort ((unsigned short) gItersArray[px][py]) == false)
        { printf ("Failure sending data...\r\n");
          return false;
        }
      }
    }
  }

  gSubNetwork.BufferedSendFlush ();

  return true;
}
