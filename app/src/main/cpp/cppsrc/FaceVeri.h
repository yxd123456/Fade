
#define FEAT_DIM 256

struct FACERC
{
	int x, y, width, height;
};

void faceVeriInit(char* faceVeriConfigPath);
int faceFeatureExtractCamera(unsigned char* pFrame, FACERC& rc, float** featc, int isCompare=false);
int faceFeatureExtractIDCard(unsigned char* pFrame, FACERC& rc, float** feati);
int faceFeatureCompare(float* score);
void faceVeriFree();
