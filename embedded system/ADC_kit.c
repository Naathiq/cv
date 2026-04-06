#include <LPC214x.h>
#include <stdio.h>
#include "delay.h"
#include "adc.h"
#include "lcd.h"

int main(void)
{
    char StrBuf[16];
    unsigned long ADC_Cnt;

    // Initialize ADC channels
    Init_ADC0(CHANNEL_1);
    Init_ADC0(CHANNEL_2);

    delay_mSec(100);

    // Initialize LCD
    LCD_Init();

    while(1)
    {
        // Read ADC Channel 1
        ADC_Cnt = Read_ADC0(CHANNEL_1);
        sprintf(StrBuf, "ADC0_CH1:%04d", ADC_Cnt);
        LCD_GoToLineOne();
        LCD_DisplayString(StrBuf);

        // Read ADC Channel 2
        ADC_Cnt = Read_ADC0(CHANNEL_2);
        sprintf(StrBuf, "ADC0_CH2:%04d", ADC_Cnt);
        LCD_GoToLineTwo();
        LCD_DisplayString(StrBuf);

        delay_mSec(500);
    }
}