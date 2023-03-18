#ifndef FRACTALSETUPDATA_H
#define FRACTALSETUPDATA_H

#define MAXSERVERS 4

class FractalSetupData
{
public:
  FractalSetupData ();
  ~FractalSetupData ();

  void CopyFromThisTo (FractalSetupData *here);

  void Load (bool defaultsOnly = false);
  void Save (void);

  // File format version
  char m_Version;
  
  // Use alternate drawing algorithm in client?
  char m_AltDraw;

  // IP addresses
  char m_LocalIP[128];
  char m_ServerIPs[MAXSERVERS][128];
  char m_UseThisServer[MAXSERVERS];

  // Networking?
  char m_BeNetworkClient;
  char m_BeNetworkServer;

  // Work division
  int m_WorkClient;
  int m_WorkServers[MAXSERVERS];

  // Autozoom specific variables
  char m_AZDrawProgress;
  char m_AZSaveImages;
  wchar_t m_SaveDir[128];
  char m_AZSaveReducedSize;

  // Screen saver specific variables
  char m_SSAutoZoom;

  // Everything necessary to describe a location.
  double m_L1MinX;
  double m_L1MinY;
  double m_L1MaxX;
  double m_L1MaxY;
  int m_L1Iterations;
  int m_L1NumFrames;

  // The algorithm desired
  int m_AlgHighRes;
  int m_AlgLowRes;

private:
  void ReadString(wchar_t *fromDisk, wchar_t *defValue, int length);
  void WriteString(wchar_t *toDisk, int length);

  void ReadString (char *fromDisk, char *defValue, int length);
  void WriteString (char *toDisk, int length);

  void ReadChar(wchar_t *fromDisk, wchar_t defValue);
  void WriteChar(wchar_t toDisk);

  void ReadChar (char *fromDisk, char defValue);
  void WriteChar (char toDisk);

  void ReadInt (int *fromDisk, int defValue);
  void WriteInt (int toDisk);

  void ReadDouble (double *fromDisk, double defValue);
  void WriteDouble (double toDisk);

  FILE *m_File;
};

#endif