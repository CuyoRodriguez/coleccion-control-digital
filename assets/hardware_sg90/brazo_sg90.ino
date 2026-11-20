// Brazo planar 2 GDL (hombro q1, codo q2) con SG90 - Arduino Robotic Arm Kit 4 SG90 Acrylic B.
// Arduino Uno: SDA=A4, SCL=A5. PCA9685: hombro CH0, codo CH1; base y pinza sin control.
// VCC del modulo a 5 V logicos; V+ a fuente externa de 5 V para servos, con GND comun.
//
// USB serial 115200: "US p1,p2\n" (us nominales) o "q1,q2\n" (grados legados).
// Respuesta: "OK US p1,p2", "OK q1,q2", "ERR" o "ERR PWM".
// "LISTO US2" se envia tras inicializar el modulo y apagar CH0/CH1.
// La primera trama habilita las salidas sin rampa; las siguientes se limitan en velocidad.
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

const byte CANAL_HOMBRO = 0;
const byte CANAL_CODO = 1;
const uint16_t PWM_HZ = 50;
Adafruit_PWMServoDriver pwm(0x40);

// Rango inicial conservador: calibrar cada servo sin carga antes de ampliar.
const uint16_t HOMBRO_MIN_US = 1000, HOMBRO_MAX_US = 2000;
const uint16_t CODO_MIN_US = 1000, CODO_MAX_US = 2000;

// ---- Calibracion: angulo_servo = OFFSET + SIGNO * q ----
// Convencion: hombro 0 = +X, 90 = +Y, 180 = -X (CCW). Codo 0 = extendido, positivo = pliega CCW.
// Si un servo gira al reves, poner SIGNO = -1 y OFFSET = 180. Probar primero sin carga.
const int OFFSET_HOMBRO = 0, SIGNO_HOMBRO = 1;
const int OFFSET_CODO = 0, SIGNO_CODO = 1;

// ---- Seguridad ----
const int Q_MIN = 5, Q_MAX = 170;       // soft limits: no forzar el acrilico contra los topes
const int US_CERO = 1000, US_MIN = 1028, US_MAX = 1944;
const long PASO_MAX = 3000L;           // 3 grados por ciclo de 20 ms, en milesimas de grado
const unsigned long PERIODO_MS = 20;

bool activo = false;                    // true tras la primera trama valida
bool pwmListo = false;
long objetivo1, objetivo2, actual1, actual2;
char buffer[20];
byte n = 0;
bool desbordado = false;
unsigned long ultimo = 0;

int limitar(int q) { return constrain(q, Q_MIN, Q_MAX); }

long aServo(long q, int offset, int signo) {
  return constrain(offset * 1000L + signo * q, 0L, 180000L);
}

long acercar(long actual, long meta) {
  if (meta > actual) return min(actual + PASO_MAX, meta);
  return max(actual - PASO_MAX, meta);
}

uint16_t pulsosPWM(long angulo, uint16_t minimoUs, uint16_t maximoUs) {
  const long anchoUs = minimoUs + (angulo * (maximoUs - minimoUs) + 90000L) / 180000L;
  return (anchoUs * 4096L * PWM_HZ + 500000L) / 1000000L;
}

bool escribirServos() {
  const uint8_t hombro = pwm.setPWM(CANAL_HOMBRO, 0, pulsosPWM(
      aServo(actual1, OFFSET_HOMBRO, SIGNO_HOMBRO), HOMBRO_MIN_US, HOMBRO_MAX_US));
  if (hombro != 0) return false;
  return pwm.setPWM(CANAL_CODO, 0, pulsosPWM(
      aServo(actual2, OFFSET_CODO, SIGNO_CODO), CODO_MIN_US, CODO_MAX_US)) == 0;
}

void falloPWM() {
  pwmListo = false;
  activo = false;
  Serial.println(F("ERR PWM"));
}

bool leerEntero(const char *s, char fin, int maximo, int &valor, const char *&sig) {
  if (*s < '0' || *s > '9') return false;
  valor = 0;
  for (; *s >= '0' && *s <= '9'; ++s) {
    long siguiente = valor * 10L + (*s - '0');
    if (siguiente > maximo) return false;
    valor = static_cast<int>(siguiente);
  }
  if (*s != fin) return false;
  sig = s + 1;
  return true;
}

void procesarLinea() {
  buffer[n] = '\0';
  const bool enUs = strncmp(buffer, "US ", 3) == 0;
  const int maximo = enUs ? US_MAX : 180;
  int a, b;
  const char *resto;
  if (!leerEntero(buffer + (enUs ? 3 : 0), ',', maximo, a, resto)
      || !leerEntero(resto, '\0', maximo, b, resto)
      || (enUs && (a < US_MIN || b < US_MIN))) {
    Serial.println(F("ERR"));
    return;
  }
  if (!pwmListo) {
    Serial.println(F("ERR PWM"));
    return;
  }
  if (!enUs) {
    a = limitar(a);
    b = limitar(b);
  }
  objetivo1 = enUs ? (a - US_CERO) * 180L : a * 1000L;
  objetivo2 = enUs ? (b - US_CERO) * 180L : b * 1000L;
  if (!activo) {
    actual1 = objetivo1;
    actual2 = objetivo2;
    if (!escribirServos()) {
      falloPWM();
      return;
    }
    ultimo = millis();
    activo = true;
  }
  Serial.print(F("OK "));
  if (enUs) Serial.print(F("US "));
  Serial.print(a);
  Serial.print(',');
  Serial.println(b);
}

void setup() {
  Serial.begin(115200);
  if (!pwm.begin()) {
    falloPWM();
    return;
  }
  if (pwm.setPWM(CANAL_HOMBRO, 0, 4096) != 0 ||
      pwm.setPWM(CANAL_CODO, 0, 4096) != 0) {
    falloPWM();
    return;
  }
  pwm.setPWMFreq(PWM_HZ);
  pwmListo = true;
  Serial.println(F("LISTO US2"));
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
    if (!escribirServos()) falloPWM();
  }
}
