#include <stdio.h>
#include <stdlib.h>

#include <endian.h>

unsigned char *read_idx3(char *filename) {
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}

	short magic;
	fread(&magic, sizeof(short), 1, fp);

	if (magic) {
		perror("File not in idx format");
		exit(EXIT_FAILURE);
	}

	unsigned char datatype, dimensions;
	fread(&datatype, sizeof(char), 1, fp);
	fread(&dimensions, sizeof(char), 1, fp);

	int n, width, height;

	fread(&n,      sizeof(int), 1, fp);
	fread(&width,  sizeof(int), 1, fp);
	fread(&height, sizeof(int), 1, fp);

	n      = be32toh(     n);
	width  = be32toh( width);
	height = be32toh(height);

	unsigned char *out = (unsigned char *) malloc(n*width*height*sizeof(unsigned char));

	fread(out, sizeof(unsigned char), n*width*height, fp);

	fclose(fp);

	return out;
}

unsigned char *read_idx1(char *filename) {
	printf("Reading list of images %s\n", filename);
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}

	short magic;
	fread(&magic, sizeof(short), 1, fp);

	if (magic) {
		perror("File not in idx format");
		exit(EXIT_FAILURE);
	}

	unsigned char datatype, dimensions;
	fread(&datatype, sizeof(char), 1, fp);
	fread(&dimensions, sizeof(char), 1, fp);

	int n;

	fread(&n, sizeof(int), 1, fp);
	n = be32toh(n);

	unsigned char *out = (unsigned char *) malloc(n*sizeof(unsigned char));

	fread(out, sizeof(unsigned char), n, fp);

	fclose(fp);

	return out;
}
