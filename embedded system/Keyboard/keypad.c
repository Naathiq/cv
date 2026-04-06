#include "LPC214x.h"
#include "keypad.h"

// Key mapping
unsigned char Key_RowCol[4][4] = {
    {'0','1','2','3'},
    {'4','5','6','7'},
    {'8','9','A','B'},
    {'C','D','E','F'}
};

void Keypad_Init(void)
{
    KEYPAD_DIR |= COL_MASK;   // Columns as output
    KEYPAD_DIR &= ~ROW_MASK;  // Rows as input
}

unsigned char Keypad_Read(void)
{
    int i, cnt = 25;

    while(cnt--)
    {
        for(i = 0; i < 4; i++)
        {
            IOSET1 |= COL_MASK;             // Set all columns HIGH
            IOCLR1 = (COL_MASK << i);       // Make one column LOW

            if((IOPIN1 & ROW1) == 0)
                return Key_RowCol[0][i];

            else if((IOPIN1 & ROW2) == 0)
                return Key_RowCol[1][i];

            else if((IOPIN1 & ROW3) == 0)
                return Key_RowCol[2][i];

            else if((IOPIN1 & ROW4) == 0)
                return Key_RowCol[3][i];
        }
    }
    return NO_KEY_PRESSED;
}