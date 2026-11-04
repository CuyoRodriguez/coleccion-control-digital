// Brazo planar 2 GDL (hombro q1, codo q2) con SG90 - Arduino Robotic Arm Kit 4 SG90 Acrylic B.
// Base y pinza no se controlan: no se les hace attach, asi que quedan sin pulso y fijas.
//
// Protocolo (USB serial, 115200 baudios): PC -> Arduino "q1,q2\n" con enteros en grados (0..180).
// Respuesta: "OK q1,q2" (angulo objetivo ya limitado a soft limits) o "ERR" si la trama es invalida.
// Al arrancar el Arduino envia "LISTO". Los servos NO reciben pulso hasta la primera trama valida;
// esa primera trama es la pose inicial (se aplica directa) y a partir de ahi el movimiento es con rampa.
#include <Servo.h>

// ---- Cableado (ajustar a como esten conectados los servos) ----
const byte PIN_HOMBRO = 9;
const byte PIN_CODO = 10;

// ---- Calibracion: angulo_servo = OFFSET + SIGNO * q ----
// Convencion: hombro 0 = +X, 90 = +Y, 180 = -X (CCW). Codo 0 = extendido, positivo = pliega CCW.
// Si un servo gira al reves, poner SIGNO = -1 y OFFSET = 180. Probar primero sin carga.
const int OFFSET_HOMBRO = 0, SIGNO_HOMBRO = 1;
const int OFFSET_CODO = 0, SIGNO_CODO = 1;

// ---- Seguridad ----
const int Q_MIN = 5, Q_MAX = 170;       // soft limits: no forzar el acrilico contra los topes
const int PASO_MAX = 3;                 // grados por ciclo de 20 ms = 150 grados/s
const unsigned long PERIODO_MS = 20;

Servo servoHombro, servoCodo;
bool activo = false;                    // true tras la primera trama valida
int objetivo1, objetivo2, actual1, actual2;
char buffer[20];
byte n = 0;
bool desbordado = false;
unsigned long ultimo = 0;

int limitar(int q) { return constrain(q, Q_MIN, Q_MAX); }

int aServo(int q, int offset, int signo) {
  return constrain(offset + signo * q, 0, 180);
}

int acercar(int actual, int meta) {
  if (meta > actual) return min(actual + PASO_MAX, meta);
  return max(actual - PASO_MAX, meta);
}

void escribirServos() {
  servoHombro.write(aServo(actual1, OFFSET_HOMBRO, SIGNO_HOMBRO));
  servoCodo.write(aServo(actual2, OFFSET_CODO, SIGNO_CODO));
}

// Entero estricto: solo digitos, rango 0..180 (lo demas es una trama corrupta).
bool leerEntero(const char *s, char fin, int &valor, const char *&sig) {
  char *e;
  long v = strtol(s, &e, 10);
  if (e == s || *e != fin || v < 0 || v > 180) return false;
  valor = (int)v;
  sig = e + 1;
  return true;
}

void procesarLinea() {
  buffer[n] = '\0';
  int a, b;
  const char *resto;
  if (!leerEntero(buffer, ',', a, resto) || !leerEntero(resto, '\0', b, resto)) {
    Serial.println(F("ERR"));
    return;
  }
  objetivo1 = limitar(a);
  objetivo2 = limitar(b);
  if (!activo) {
    actual1 = objetivo1;
    actual2 = objetivo2;
    escribirServos();
    servoHombro.attach(PIN_HOMBRO);
    servoCodo.attach(PIN_CODO);
    escribirServos();
    ultimo = millis();
    activo = true;
  }
  Serial.print(F("OK "));
  Serial.print(objetivo1);
  Serial.print(',');
  Serial.println(objetivo2);
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("LISTO"));
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      if (desbordado) Serial.println(F("ERR"));
      else if (n > 0) procesarLinea();
      n = 0;
      desbordado = false;
    } else if (c != '\r') {
      if (n < sizeof(buffer) - 1) buffer[n++] = c;
      else desbordado = true;  // trama demasiado larga: se descarta entera al llegar el '\n'
    }
  }
  if (activo && millis() - ultimo >= PERIODO_MS) {
    ultimo += PERIODO_MS;
    if (millis() - ultimo >= PERIODO_MS) ultimo = millis();  // no acumular retraso
    actual1 = acercar(actual1, objetivo1);
    actual2 = acercar(actual2, objetivo2);
    escribirServos();
  }
}
