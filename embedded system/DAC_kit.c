#include <LPC214x.h>

#define DAC_BIAS 0x00010000

// Function to initialize DAC pin
void DACInit(void)
{
    // Clear bits and select DAC function (P0.25)
    PINSEL1 &= 0xFF3FFFFF;
    PINSEL1 |= 0x00080000;
}

// Simple delay function
void delay(unsigned int d)
{
    unsigned int i;
    for(i = 0; i < d; i++);
}

int main(void)
{
    unsigned int i;

    DACInit();  // Initialize DAC

    while(1)
    {
        // Rising slope
        for(i = 0; i < 1024; i++)
        {
            DACR = (i << 6) | DAC_BIAS;
            delay(500);
        }

        // Falling slope
        for(i = 1023; i > 0; i--)
        {
            DACR = (i << 6) | DAC_BIAS;
            delay(500);
        }
    }
}
