import serial
import time
import sys
import os

def send_gcode_file(port, baudrate, gcode_file):
    """
    Sends a G-code file to a GRBL controller over serial.
    """
    # Validate file existence
    if not os.path.isfile(gcode_file):
        print(f"Error: File '{gcode_file}' not found.")
        return

    try:
        # Open serial connection
        with serial.Serial(port, baudrate, timeout=1) as grbl:
            print(f"Connected to GRBL on {port} at {baudrate} baud.")

            # Wake up GRBL
            grbl.write(b"\r\n\r\n")
            time.sleep(2)  # Wait for GRBL to initialize
            grbl.flushInput()  # Clear startup text

            # Open G-code file
            with open(gcode_file, 'r') as f:
                for line in f:
                    # Strip comments and whitespace
                    clean_line = line.strip()
                    if clean_line == "" or clean_line.startswith("("):
                        continue  # Skip empty/comment lines

                    # Send line to GRBL
                    grbl.write((clean_line + '\n').encode('utf-8'))

                    # Wait for GRBL response
                    grbl_response = grbl.readline().decode('utf-8').strip()
                    while grbl_response == "":
                        grbl_response = grbl.readline().decode('utf-8').strip()

                    print(f"Sent: {clean_line} | Response: {grbl_response}")

                    # Optional: small delay to avoid buffer overflow
                    time.sleep(0.01)

            print("G-code upload complete.")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Example usage — change these values for your setup
    SERIAL_PORT = "COM4"       # e.g., "COM3" on Windows or "/dev/ttyUSB0" on Linux
    BAUD_RATE = 115200         # Default GRBL baud rate
    GCODE_FILE = "alpana2_scaled.gcode"

    send_gcode_file(SERIAL_PORT, BAUD_RATE, GCODE_FILE)
