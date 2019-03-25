#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
char * key = "alibaba"; // 秘钥，这是加密的关键，方法很简单

int encryptFile(char *sourcefile, char *secretKey, char *targetFile){
	FILE *fpSource, *fpTarget;  // 要打开的文件的指针
	char buffer[21];  // 缓冲区，用于存放从文件读取的数据
	int readCount,  // 每次从文件中读取的字节数
		keyLen = strlen(secretKey),  // 密钥的长度
		i;  // 循环次数
 
	// 以二进制方式读取/写入文件
	fpSource = fopen(sourcefile, "rb");
	if(fpSource==NULL){
		printf("文件[%s]打开失败，请检查文件路径和名称是否输入正确！\n", sourcefile);
		return 0;
	}
	fpTarget = fopen(targetFile, "wb");
	if(fpTarget==NULL){
		printf("文件[%s]创建/写入失败！请检查文件路径和名称是否输入正确！\n", targetFile);
		return 0;
	}
 
	// 不断地从文件中读取 keyLen 长度的数据，保存到buffer，直到文件结束
	while( (readCount=fread(buffer, 1, keyLen, fpSource)) > 0 ){
		// 对buffer中的数据逐字节进行异或运算
		for(i=0; i<readCount; i++){
			buffer[i] ^= secretKey[i];
		}
		// 将buffer中的数据写入文件
		fwrite(buffer, 1, readCount, fpTarget);
	}
 
	fclose(fpSource);
	fclose(fpTarget);
 
	return 1;
}

int main(int argc, char **argv)
{ 
    encryptFile(argv[1],key,argv[2]);
    /*
    FILE *inputFile;
    inputFile = fopen(argv[1], "rb");
    if (!inputFile) {
        fprintf(stderr, "Can't open bin file!");
        exit(1);
    }

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, inputFile);
    fread(&minor, sizeof(int), 1, inputFile);
    fread(&revision, sizeof(int), 1, inputFile);
    printf("%d %d %d",major,minor,revision);
    major = 

    fseek(inputFile, 0, SEEK_END);
    long inputFileLength = ftell(inputFile);
    printf("input file length: %ld\n", inputFileLength);
    fseek(inputFile, 0, SEEK_SET);

    fclose(inputFile);
    */


}
