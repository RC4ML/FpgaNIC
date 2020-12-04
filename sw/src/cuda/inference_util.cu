#include"inference_util.cuh"
#include<vector>
#include <string>
#include <dirent.h>
#include <iostream>
#include <fstream>

using namespace std;

int getFileList(string dirent, vector<string> &FileList){
    DIR *p_dir;
    struct dirent *p_dirent;

    if((p_dir = opendir((dirent).c_str())) == NULL){
        cout << "check pir path:" << (dirent).c_str() << "failed" <<endl;
        return -1;
    }
    while((p_dirent=readdir(p_dir)))
    {
        string s(p_dirent->d_name);
        if(s != "." && s != "..")
            FileList.push_back(s);
    }
    closedir(p_dir);
    return FileList.size();
}

void load_data(float data[][150528]){
	string path = "/home/amax6/cj/sw/create_model/data/";
	vector<string> fileList;
	int ret = getFileList(path, fileList);
    if(ret < 0)
        cout << "read data filed"<< endl;
    cout << "fileList:" << endl;
	int img_count=0;
    for(auto imageName : fileList){
        string imagePath = path + imageName;
        cout << imagePath << endl;
		ifstream readFile(imagePath);
		for(int j=0;j<150528;j++){
			readFile>>data[img_count][j];
		}
		img_count++;
    }
	cout << "load data done!\n" << endl;
}

__global__ void get_max(float* data){
	float max_value=-999;
	int max_index=0;
	for(int i=0;i<1000;i++){
		if(data[i]>max_value){
			max_value=data[i];
			max_index=i;
		}
	}
	printf("max index:%d  max value:%f\n",max_index,max_value);
}