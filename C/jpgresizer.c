#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

int main(int argc, char **argv)
{
	printf("jpgresizer\n");

	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;


	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);

	exit(0);
}


