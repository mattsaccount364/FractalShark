// FractalServer2.cpp : Defines the entry point for the application.
//

// Make this program rely on Fractal for everything.

#include "stdafx.h"
#include "resource.h"

#include <winsock.h>
#include <stdio.h>
#include <conio.h>

#include "..\Fractal.h"
#include "..\FractalNetwork.h"
#include "..\FractalSetupData.h"

FractalNetwork *gExitNetwork;

wchar_t gOutputMessage[512];
Fractal *gFractal = NULL;

void Initialize (void);
void Uninitialize (void);

void OutputMessage(const wchar_t *szFormat, ...);
void OutputMessageAndErase (int deleteXChars, const wchar_t *szFormat, ...);
void DeleteXChars (int x);
void ClearOutput (void);

// Global Variables:
HINSTANCE hInst;                // current instance
const wchar_t *szWindowClass = L"FractalServerClass";

// Foward declarations of functions included in this code module:
ATOM        MyRegisterClass (HINSTANCE hInstance);
BOOL        InitInstance (HINSTANCE, int);
LRESULT CALLBACK  WndProc (HWND, UINT, WPARAM, LPARAM);

HWND gHWnd;

int APIENTRY WinMain (HINSTANCE hInstance,
                     HINSTANCE /*hPrevInstance*/,
                     LPSTR     /*lpCmdLine*/,
                     int       nCmdShow)
{ MSG msg;

  // Initialize global strings
  MyRegisterClass (hInstance);

  // Perform application initialization:
  if (!InitInstance (hInstance, nCmdShow))
  { return FALSE; }

  Initialize ();

  // Main message loop:
  while (GetMessage (&msg, NULL, 0, 0))
  { TranslateMessage (&msg);
    DispatchMessage (&msg);
  }

  Uninitialize ();

  return msg.wParam;
}

void Initialize (void)
{ // Initialize the output string.
  gOutputMessage[0] = 0;

  // Initialize Winsock
  WSADATA info;
  if (WSAStartup (MAKEWORD (1, 1), &info) != 0)
  { MessageBox (NULL, L"Cannot initialize WinSock!", L"WSAStartup", MB_OK);
    return;
  }

  // Create the fractal
  FractalSetupData setupData;
  setupData.Load ();
  setupData.m_BeNetworkServer = 'y'; // We ARE the network server.
  setupData.m_BeNetworkClient = 'n';
  gFractal = new Fractal (&setupData, 5, 5, OutputMessage, NULL, false);
}

void Uninitialize (void)
{ if (gFractal != NULL)
  { delete gFractal; }

  WSACleanup ();
}

//
//  FUNCTION: MyRegisterClass ()
//
//  PURPOSE: Registers the window class.
//
//  COMMENTS:
//
//    This function and its usage is only necessary if you want this code
//    to be compatible with Win32 systems prior to the 'RegisterClassEx'
//    function that was added to Windows 95. It is important to call this function
//    so that the application will get 'well formed' small icons associated
//    with it.
//
ATOM MyRegisterClass (HINSTANCE hInstance)
{ WNDCLASSEX wcex;

  wcex.cbSize = sizeof (WNDCLASSEX);

  wcex.style      = CS_HREDRAW | CS_VREDRAW;
  wcex.lpfnWndProc  = (WNDPROC) WndProc;
  wcex.cbClsExtra    = 0;
  wcex.cbWndExtra    = 0;
  wcex.hInstance    = hInstance;
  wcex.hIcon      = LoadIcon (hInstance, (LPCTSTR) IDI_FRACTALSERVER2);
  wcex.hCursor    = LoadCursor (NULL, IDC_ARROW);
  wcex.hbrBackground  = (HBRUSH) GetStockObject (WHITE_BRUSH);
  wcex.lpszMenuName  = (LPCWSTR) IDC_FRACTALSERVER2;
  wcex.lpszClassName  = szWindowClass;
  wcex.hIconSm    = LoadIcon (wcex.hInstance, (LPCTSTR) IDI_SMALL);

  return RegisterClassEx (&wcex);
}

//
//   FUNCTION: InitInstance (HANDLE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance (HINSTANCE hInstance, int nCmdShow)
{ HWND hWnd;

  hInst = hInstance; // Store instance handle in our global variable

  hWnd = CreateWindow (szWindowClass, L"Fractal Server", WS_OVERLAPPEDWINDOW,
     CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);

  if (!hWnd)
  { return FALSE; }

  ShowWindow (hWnd, nCmdShow);
  UpdateWindow (hWnd);

  gHWnd = hWnd;
  return TRUE;
}

//
//  FUNCTION: WndProc (HWND, unsigned, WORD, LONG)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT  - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc (HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  int wmId, wmEvent;
  PAINTSTRUCT ps;
  HDC hdc;

  switch (message)
  {
    case WM_COMMAND:
      wmId    = LOWORD (wParam);
      wmEvent = HIWORD (wParam);
      // Parse the menu selections:
      switch (wmId)
      {
        case IDM_EXIT:
           DestroyWindow (hWnd);
           break;
        default:
           return DefWindowProc (hWnd, message, wParam, lParam);
      }
      break;
    case WM_PAINT:
      hdc = BeginPaint (hWnd, &ps);

      RECT dimensions;
      GetClientRect (hWnd, &dimensions);

      FillRect (hdc, &dimensions, (HBRUSH) GetStockObject (WHITE_BRUSH));

      SetTextColor (hdc, RGB (0, 0, 0));
      DrawText (hdc, gOutputMessage, wcslen (gOutputMessage), &dimensions, DT_LEFT);
      EndPaint (hWnd, &ps);
      break;
    case WM_DESTROY:
      PostQuitMessage (0);
      break;
    default:
      return DefWindowProc (hWnd, message, wParam, lParam);
   }
   return 0;
}

void OutputMessage (const wchar_t *szFormat, ...)
{ CRITICAL_SECTION cs;
  InitializeCriticalSection (&cs);
  EnterCriticalSection (&cs);

  va_list argList;
  va_start (argList, szFormat);

  wchar_t newMessage[512];
  vswprintf (newMessage, szFormat, argList);

  if (wcslen (newMessage) + wcslen (gOutputMessage) >= 512)
  { memmove (gOutputMessage, gOutputMessage + wcslen (newMessage), wcslen (gOutputMessage));
    gOutputMessage[511 - wcslen (newMessage)] = 0;
  }

  int backUp = 0;
  for (int i = wcslen (newMessage) - 1; i >= 0; i--)
  { if (newMessage[i] == '\b')
    { backUp++;
      newMessage[i] = 0;
    }
    else
    { break; }
  }

  wcscat (gOutputMessage, newMessage);

  va_end (argList);

  RedrawWindow (gHWnd, NULL, NULL, RDW_INVALIDATE);

  int end = wcslen (gOutputMessage) - backUp;
  if (end >= 0)
  { gOutputMessage[end] = 0; }

  LeaveCriticalSection (&cs);
  DeleteCriticalSection (&cs);
}

void DeleteXChars (int x)
{ static volatile bool working;
  while (working == true)
  { Sleep (100); }
  working = true;

  int end = wcslen (gOutputMessage) - x;
  if (end >= 0)
  { gOutputMessage[end] = 0; }

  working = false;
}

void ClearOutput (void)
{ gOutputMessage[0] = 0; }
