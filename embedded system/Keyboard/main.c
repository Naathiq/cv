#include "LPC214x.h"
#include "lcd.h"
#include "keypad.h"
#include "delay.h"

int main(void)
{
    unsigned char Key;

    LCD_Init();
    Keypad_Init();

    LCD_GoToLineOne();
    LCD_DisplayString("ARM LPC2148");

    delay_mSec(2000);

    while(1)
    {
        Key = Keypad_Read();

        if(Key != NO_KEY_PRESSED)
        {
            LCD_GoToLineTwo();
            LCD_DisplayString("Key ");
            LCD_DataWrite(Key);
            LCD_DisplayString(" is Pressed");

            delay_mSec(1000);

            LCD_ClearLineTwo();
        }
    }
}