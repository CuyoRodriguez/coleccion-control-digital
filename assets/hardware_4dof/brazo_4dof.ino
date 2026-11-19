// Prototipo de protocolo 4 GDL para Arduino Uno. NO mueve servos.
// Los pines son una propuesta sin verificar, no se asignan a los servos.
// Recibe cuatro grados enteros (0..180), separados por comas y terminados en \n.
// La respuesta SIM confirma solo el analisis de la trama, NO una posicion real.
#include <Servo.h>
#include <stdlib.h>

Servo motores[4];  // Sin attach ni write: las cuatro salidas permanecen deshabilitadas.
const int PINES_PROPUESTOS[4] = {-1, -1, -1, -1};
char buffer[32];
byte longitud = 0;
bool desbordado = false;

bool analizar(int valores[4]) {
  const char *inicio = buffer;
  for (int i = 0; i < 4; ++i) {
    char *fin;
    if (*inicio < '0' || *inicio > '9') return false;
    long valor = strtol(inicio, &fin, 10);
    if (valor < 0 || valor > 180 || *fin != (i == 3 ? '\0' : ',')) return false;
    valores[i] = (int)valor;
    inicio = fin + 1;
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("BLOQUEADO: calibracion y pines pendientes"));
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      buffer[longitud] = '\0';
      int valores[4];
      if (desbordado || !analizar(valores)) {
        Serial.println(F("ERR"));
      } else {
        Serial.print(F("SIM "));
        for (int i = 0; i < 4; ++i) {
          if (i) Serial.print(',');
          Serial.print(valores[i]);
        }
        Serial.println(F(" - servos deshabilitados"));
      }
      longitud = 0;
      desbordado = false;
    } else if (c != '\r') {
      if (longitud < sizeof(buffer) - 1) buffer[longitud++] = c;
      else desbordado = true;
    }
  }
}
