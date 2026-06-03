echo "This is for running vcxsrv on Windows and having it work with FractalShark"
echo "Example use is build on Linux, run GUI on Windows"

& 'C:\Program Files\VcXsrv\vcxsrv.exe' :0 -multiwindow -clipboard +iglx -wgl

echo "Start ssh like e.g. ssh -X -Y matthew@localhost"