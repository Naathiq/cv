#include <stdio.h> 
#include "LPC214X.h" 
void delay (void); 
int main() 
{ while (1) 
{ 
IODIR0 = (1<<4) | (0 << 5) |(1<<6) | (0 << 7) | (1<<10) | (0 << 11) | (1<<12) | (0 << 13); 
delay(); 
IODIR0 = (0<<4) | (0 << 5) |(1<<6) | (1 << 7) | (0<<10) | (0 << 11) | (1<<12) | (1 << 13); 
delay(); 
} 
} 
void delay (void) 
{ 
unsigned int i; 
for (i = 0; i < 2500000; i++); 
}