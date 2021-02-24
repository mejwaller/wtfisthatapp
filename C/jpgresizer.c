#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

/*Directory parsing funtions*/
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

const char* inpath="..//dunpics//";


int main(int argc, char **argv)
{
	int i;
	DIR *indir;
	FILE *infile,*outfile;
	struct dirent *entry;

	printf("jpgresizer\n");
	printf("inpath: %s\n",inpath);

	if((indir=opendir(inpath))==NULL) {
		fprintf(stderr,"Cannot open dierctory: %s\n",inpath);
		exit(1);
	}

	chdir(inpath);
	
	while((entry=readdir(indir)) != NULL) {
	       if(strcmp(".",entry->d_name) == 0 || strcmp("..",entry->d_name)==0) 
		       continue;

	       struct jpeg_decompress_struct cinfo;
	       struct jpeg_error_mgr jerr;
	       int height;

	       cinfo.err = jpeg_std_error(&jerr);
	       jpeg_create_decompress(&cinfo);

	       printf("Opening file %s\n",entry->d_name);
	       if((infile = fopen(entry->d_name,"rb"))==NULL) {
		       fprintf(stderr,"Can't open %s\n", entry->d_name);
		       exit(1);
	       }
	       jpeg_stdio_src(&cinfo,infile);
	       jpeg_read_header(&cinfo, TRUE);
	       jpeg_start_decompress(&cinfo);

	       /*printf("output_withd: %d\n",cinfo.output_width);
	       printf("Output components: %d\n",cinfo.output_components);
	       printf("output_height: %d\n", cinfo.output_height);
	       */

	       height= cinfo.output_height;

	       JSAMPARRAY lines = (JSAMPARRAY)malloc(cinfo.output_height*sizeof(JSAMPROW));
	       for(i=0; i<cinfo.output_height; i++) 
		       lines[i] = (JSAMPROW)malloc(cinfo.output_width*cinfo.output_components*sizeof(JSAMPLE));

	      /*printf("reading in jpeg...\n");*/

	      while(cinfo.output_scanline < cinfo.output_height) {
		      /*printf("(readlines = %d, lines to read = %d\n",cinfo.output_scanline, cinfo.output_height);*/
		      jpeg_read_scanlines(&cinfo,lines,cinfo.output_height);
	      }

	      printf("Finishing decompress...\n");
	      jpeg_finish_decompress(&cinfo);

	      printf("Cleainign up...\n");
	      jpeg_destroy((j_common_ptr)&cinfo);

	      /*do resizing*/

	      printf("Freeing memory...\n");

	      for(i=0; i< height; i++)
		      free(lines[i]);
	      free(lines);

	      fclose(infile);
	}


	chdir("..");
	closedir(indir);	

	exit(0);
}


