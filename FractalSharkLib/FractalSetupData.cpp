#include "stdafx.h"
#include "FractalSetupData.h"

#include <stdio.h>

// Set to some constant greater than 0.
// Defines the file version.
const unsigned char VERSION = 9;

FractalSetupData::FractalSetupData ()
{
}

FractalSetupData::~FractalSetupData ()
{
}

void FractalSetupData::CopyFromThisTo (FractalSetupData *here)
{ int i;
  here->m_Version = m_Version;
  
  strcpy (here->m_LocalIP, m_LocalIP);
  for (i = 0; i < MAXSERVERS; i++)
  { strcpy (here->m_ServerIPs[i], m_ServerIPs[i]); }
  for (i = 0; i < MAXSERVERS; i++)
  { here->m_UseThisServer[i] = m_UseThisServer[i]; }

  here->m_BeNetworkClient = m_BeNetworkClient;
  here->m_BeNetworkServer = m_BeNetworkServer;

  here->m_WorkClient = m_WorkClient;
  for (i = 0; i < MAXSERVERS; i++)
  { here->m_WorkServers[i] = m_WorkServers[i]; }

  here->m_AZDrawProgress = m_AZDrawProgress;
  here->m_AZSaveImages = m_AZSaveImages;
  wcscpy (here->m_SaveDir, m_SaveDir);
  here->m_AZSaveReducedSize = m_AZSaveReducedSize;

  here->m_SSAutoZoom = m_SSAutoZoom;

  here->m_L1MinX = m_L1MinX;
  here->m_L1MinY = m_L1MinY;
  here->m_L1MaxX = m_L1MaxX;
  here->m_L1MaxY = m_L1MaxY;
  here->m_L1Iterations = m_L1Iterations;
  here->m_L1NumFrames = m_L1NumFrames;
}

void FractalSetupData::Load (bool defaultsOnly)
{ // Load and save the INI file to the C:\\Windows directory.
  wchar_t FileName[512];

  if (defaultsOnly == false)
  { wchar_t WindowsDir[256];
    wsprintf(WindowsDir, L"C:\\Fractal Saves");

    wcscpy (FileName, WindowsDir);
    if (WindowsDir[wcslen (WindowsDir) - 1] != '\\')
    { wcscat (FileName, L"\\"); }

    wcscat (FileName, L"!FractalSetup.ini");
    m_File = _wfopen (FileName, L"rb+");
  }
  else
  { m_File = NULL; }

  ReadChar (&m_Version, 0);
  if (m_Version != VERSION)
  { if (m_File != NULL)
    { fclose (m_File);
      m_File = NULL;
      _wunlink (FileName);
    }
  }

  int i;
  
  ReadString (m_LocalIP, "192.168.0.2", 128);
  for (i = 0; i < MAXSERVERS; i++)
  { ReadString (m_ServerIPs[i], "192.168.0.1", 128); }
  for (i = 0; i < MAXSERVERS; i++)
  { ReadChar (&m_UseThisServer[i], 'n'); }

  ReadChar (&m_BeNetworkClient, 'n');
  ReadChar (&m_BeNetworkServer, 'n');

  ReadInt (&m_WorkClient, 20);
  for (i = 0; i < MAXSERVERS; i++)
  { ReadInt (&m_WorkServers[i], 20); }

  ReadChar (&m_AZDrawProgress, 'y');
  ReadChar (&m_AZSaveImages, 'y');
  ReadString (m_SaveDir, L"C:\\Fractal Saves", 128);
  ReadChar (&m_AZSaveReducedSize, 'y');

  ReadChar (&m_SSAutoZoom, 'n');

  ReadDouble (&m_L1MinX, -2.5);
  ReadDouble (&m_L1MinY, -1.5);
  ReadDouble (&m_L1MaxX, 1.5);
  ReadDouble (&m_L1MaxY, 1.5);
  ReadInt (&m_L1Iterations, 256);
  ReadInt (&m_L1NumFrames, 1500);
  
  if (m_File != NULL)
  { fclose (m_File); }
}

void FractalSetupData::Save (void)
{ { wchar_t WindowsDir[256];
    wsprintf(WindowsDir, L"C:\\Fractal Saves");
  
    wchar_t FileName[512];
    wcscpy (FileName, WindowsDir);
    if (WindowsDir[wcslen (WindowsDir) - 1] != '\\')
    { wcscat (FileName, L"\\"); }
  
    wcscat (FileName, L"!FractalSetup.ini");
    m_File = _wfopen (FileName, L"wb+");
  }

  int i;

  WriteChar ((char) VERSION);
  
  WriteString (m_LocalIP, 128);
  for (i = 0; i < MAXSERVERS; i++)
  { WriteString (m_ServerIPs[i], 128); }
  for (i = 0; i < MAXSERVERS; i++)
  { WriteChar (m_UseThisServer[i]); }

  WriteChar (m_BeNetworkClient);
  WriteChar (m_BeNetworkServer);

  WriteInt (m_WorkClient);
  for (i = 0; i < MAXSERVERS; i++)
  { WriteInt (m_WorkServers[i]); }

  WriteChar (m_AZDrawProgress);
  WriteChar (m_AZSaveImages);
  WriteString (m_SaveDir, 128);
  WriteChar (m_AZSaveReducedSize);

  WriteChar (m_SSAutoZoom);

  WriteDouble (m_L1MinX);
  WriteDouble (m_L1MinY);
  WriteDouble (m_L1MaxX);
  WriteDouble (m_L1MaxY);
  WriteInt (m_L1Iterations);
  WriteInt (m_L1NumFrames);

  fclose (m_File);
}

void FractalSetupData::ReadString(wchar_t *fromDisk, const wchar_t *defValue, int length)
{
    int i;
    if (m_File == NULL)
    {
        for (i = 0; i < length; i++)
        {
            fromDisk[i] = defValue[i];
        }
    }
    else
    {
        for (i = 0; i < length; i++)
        {
            ReadChar(&fromDisk[i], defValue[i]);
        }
    }
}


void FractalSetupData::ReadString (char *fromDisk, const char *defValue, int length)
{ int i;
  if (m_File == NULL)
  { for (i = 0; i < length; i++)
    { fromDisk[i] = defValue[i]; }
  }
  else
  { for (i = 0; i < length; i++)
    { ReadChar (&fromDisk[i], defValue[i]); }
  }
}

void FractalSetupData::WriteString (char *toDisk, int length)
{ int i;
  for (i = 0; i < length; i++)
  { WriteChar (toDisk[i]); }
}

void FractalSetupData::WriteString(wchar_t *toDisk, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        WriteChar(toDisk[i]);
    }
}

void FractalSetupData::ReadChar (char *fromDisk, const char defValue)
{ if (m_File == NULL)
  { *fromDisk = defValue; }
  else
  { fread (fromDisk, sizeof (char), 1, m_File); }
}

void FractalSetupData::ReadChar(wchar_t *fromDisk, const wchar_t defValue)
{
    char temp;
    if (m_File == NULL)
    {
        *fromDisk = defValue;
    }
    else
    {
        fread(&temp, sizeof(char), 1, m_File);
        *fromDisk = temp;
    }
}


void FractalSetupData::WriteChar (char toDisk)
{ if (m_File != NULL)
  { fwrite (&toDisk, sizeof (char), 1, m_File); }
}

void FractalSetupData::WriteChar(wchar_t toDisk)
{
    if (m_File != NULL)
    {
        char temp = (char)toDisk;
        fwrite(&temp, sizeof(char), 1, m_File);
    }
}

void FractalSetupData::ReadInt (int *fromDisk, const int defValue)
{ if (m_File == NULL)
  { *fromDisk = defValue; }
  else
  { fread (fromDisk, sizeof (int), 1, m_File); }
}

void FractalSetupData::WriteInt (int toDisk)
{ if (m_File != NULL)
  { fwrite (&toDisk, sizeof (int), 1, m_File); }
}

void FractalSetupData::ReadDouble (double *fromDisk, const double defValue)
{ if (m_File == NULL)
  { *fromDisk = defValue; }
  else
  { fread (fromDisk, sizeof (double), 1, m_File); }
}

void FractalSetupData::WriteDouble (double toDisk)
{ if (m_File != NULL)
  { fwrite (&toDisk, sizeof (double), 1, m_File); }
}