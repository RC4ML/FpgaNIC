#include<iostream>
using namespace std;
struct __attribute__((__visibility__("default"))) TEST {
	void record() {
    fail();
  }

public:
  void fail() {
    cout<<"test";
  }
};
int main(){
	TEST a;
	a.fail();
	return 0;
}