/* 
LM35 analog temperature sensor with Arduino example code. 
More info: https://www.makerguides.com
*/

// Define pin connections
int tempsensor = A0;   // LM35 sensor pin
int pi = 2;            // Output control pin
int green = 3;         // Green LED pin
int red = 4;           // Red LED pin

// State variables
int flash = 0;

void setup() {
  // Begin serial communication at a baud rate of 9600
  Serial.begin(9600);
  
  // Set pin modes
  pinMode(tempsensor, INPUT);
  pinMode(pi, OUTPUT);
  pinMode(green, OUTPUT);
  pinMode(red, OUTPUT);
  
  // Ensure initial state is LOW
  digitalWrite(pi, LOW);
}

void loop() {
  // Get a reading from the temperature sensor
  int reading = analogRead(tempsensor);
  
  // Convert reading to voltage (millivolts)
  float voltage = reading * (5000 / 1024.0);
  
  // Convert voltage to temperature in Celsius
  float temperature = voltage / 10.0;
  
  // Print the temperature in the Serial Monitor
  Serial.print(temperature);
  Serial.print(" \xC2\xB0"); // Degree symbol
  Serial.println("C");

  delay(1000); // Wait a second between readings

  // Temperature thresholds
  if (temperature > 45) {
    digitalWrite(pi, HIGH);
    digitalWrite(red, HIGH);
    digitalWrite(green, LOW);
    Serial.println("PI - HIGH");
  }
  else if (temperature < 25) {
    digitalWrite(pi, LOW);
    digitalWrite(red, LOW);
    digitalWrite(green, HIGH);
    Serial.println("PI - LOW");
  }
  else if (temperature > 25 && temperature <= 45) {
    // Flashing behavior between 25 and 45 degrees Celsius
    if (flash == 0) {
      digitalWrite(red, LOW);
      digitalWrite(green, HIGH);
      flash = 1;
    }
    else {
      digitalWrite(red, HIGH);
      digitalWrite(green, LOW);
      flash = 0;
    }
    delay(1000); // Delay for flashing effect
  }
}
