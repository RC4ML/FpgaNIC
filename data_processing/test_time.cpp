#include<cstdlib>
#include<ctime>
#include<iostream>
using namespace std;
int main(){
   clock_t t1,t2;
   t1=clock();
   /*程序待测时间部分*/
   t2=clock();
   cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<endl;
}