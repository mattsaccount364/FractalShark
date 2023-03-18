#ifndef CBITMAPWRITER_H
#define CBITMAPWRITER_H

class CBitmapWriter
{
public:
  CBitmapWriter ();
  ~CBitmapWriter ();

  void WritePng(const wchar_t *filename, unsigned char *data, int width, int height);
  int Write(const wchar_t *filename, unsigned char *data, int width, int height);
  int write24BitBmpFile(const wchar_t *filename, unsigned int width, unsigned int height, unsigned char *image);
  
  int beginWrite (const wchar_t *filename, int width, int height);
  int writeLines (unsigned char *data, int width, int numLines);
  void endWrite (void);
  
private:
  FILE *file;
};

#endif