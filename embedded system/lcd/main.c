#include <stdio.h>
#include "GPIO.h"
#include "LCD.h"
#include "LPC213x.h"

int main(void)
{
    int i;

    unsigned char line1[] = "ECE ";
    unsigned char line2[] = "2027";

    initLCD();   // Initialize LCD

    // Display first line
    for(i = 0; line1[i] != '\0'; i++)
    {
        LCDdata(line1[i]);
    }

    // Move cursor to second line
    LCDcommand(0xC0);

    // Display second line
    for(i = 0; line2[i] != '\0'; i++)
    {
        LCDdata(line2[i]);
    }

    while(1);   // Infinite loop
}