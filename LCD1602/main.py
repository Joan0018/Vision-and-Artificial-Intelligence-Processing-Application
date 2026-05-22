from lcd1602 import *

if __name__ == '__main__':
    try:
        lcd_init()
        while True:
            lcd_string("Pi 5 Trixie", LINE_1)
            lcd_string("Testing LCD1602", LINE_2)
            time.sleep(3)
            lcd_string("Direct I2C", LINE_1)
            lcd_string("ADDR = 0x27", LINE_2)
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        lcd_byte(0x01, LCD_CMD)