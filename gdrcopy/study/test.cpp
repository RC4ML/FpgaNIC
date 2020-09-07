#include <unistd.h>
#include <stdio.h>

int main(){
    printf("_SC_ARG_MAX=%ld; ARG_MAX=%ld\n", _SC_ARG_MAX, sysconf(_SC_PAGESIZE));
    printf("_SC_OPEN_MAX=%ld; OPEN_MAX=%ld\n", _SC_OPEN_MAX, sysconf(_SC_OPEN_MAX));
    return 0;
}