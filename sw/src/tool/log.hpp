#ifndef _LOG_HPP
#define _LOG_HPP
#include <stdio.h>

#define LOG_INFO 0
#define LOG_DEBUG 1
#define LOG_PRINT 1
#define LOG_ERROR 1
#define SLOW_DEBUG 0

#if LOG_ERROR
#define cjerror(...)	printf(__VA_ARGS__);
						
#else
#define cjerror(...)
#endif



#if LOG_PRINT
#define cjprint(...)	printf(__VA_ARGS__);		
#else
#define cjprint(...) 
#endif



#if LOG_DEBUG
#define cjdebug(...) 	printf(__VA_ARGS__);
						
#else
#define cjdebug(...)
#endif



#if LOG_INFO
#define cjinfo(...) printf(__VA_ARGS__)
#else
#define cjinfo(...)
#endif

#endif