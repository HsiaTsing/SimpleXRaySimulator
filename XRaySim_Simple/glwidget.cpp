#include <QtGui>
#include "gl\glew.h"
#include <QGLWidget>
#include "glwidget.h"

#include <gl/glut.h>
#define GL_PI 3.1415926
#define GL_RADIUX  0.2f
#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

Model model;

float rotate_angle1 =0;
float rotate_angle2 =0;

extern float xMRot;
extern GLuint g_XRayTexID;


GLWidget::GLWidget(QWidget *parent) : QGLWidget(QGLFormat(QGL::SampleBuffers),parent)
{

	xRot = -800;
	yRot = -800;
	zRot = 0;
	zoom = 10;



	setFocusPolicy(Qt::StrongFocus);

}

GLWidget::~GLWidget()
{
}

QSize GLWidget::minimumSizeHint() const
{
	return QSize(50, 50);
}

QSize GLWidget::sizeHint() const
{
	return QSize(512, 512);
}

static void qNormalizeAngle(int &angle)
{
	while (angle < 0)
		angle += 360 * 16;
	while (angle > 360 * 16)
		angle -= 360 * 16;
}

void GLWidget::setXRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != xRot) {
		xRot = angle;
		emit xRotationChanged(angle);
		//updateGL();
	}
}

void GLWidget::setYRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != yRot) {
		yRot = angle;
		emit yRotationChanged(angle);
		//updateGL();	
	}
}

void GLWidget::setZRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != zRot) {
		zRot = angle;
		emit zRotationChanged(angle);
		//updateGL();	
	}
}


void GLWidget::initializeGL()
{
	glClearColor(0.0,0.0,0.0,0.0);
	glClearDepth(1.0);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	model.load("testObject.obj",1.0,true,false);
	model.init();

}


void GLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor3f(1.0,1.0,1.0);

	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();
	gluLookAt(0,zoom,0,0,0,0,0,0,-1);

	glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
	glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
	glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);

	glLineWidth(2.0f);
	glBegin(GL_LINES);
		glColor3f(1.0,0.0,0.0);
		glVertex3f(0,0,0);
		glVertex3f(100,0,0);
		glColor3f(0.0,1.0,0.0);
		glVertex3f(0,0,0);
		glVertex3f(0,100,0);
		glColor3f(0.0,0.0,1.0);
		glVertex3f(0,0,0);
		glVertex3f(0,0,100);
	glEnd();
	glLineWidth(1.0f);
	glColor4f(0.5,0.5,0.5,0.5);
	glBegin(GL_LINES);
	for(int i =0 ;i< 400; i++)
	{
		float offset = 0.5*(float)i;
		glVertex3f(-100+offset,0,100);
		glVertex3f(-100+offset,0,-100);
		glVertex3f(-100,0,-100+offset);
		glVertex3f(100,0,-100+offset);
	}
	glEnd();

	glColor4f(1.0,1.0,1.0,0.4);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	float lightpos[] = { 0.f, 100.f, 150.f, 1.f};
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
	glEnable(GL_DEPTH_TEST);

	glPushMatrix();

	glRotatef(-xMRot*180.0/GL_PI,1.0,0.0,0.0);

	model.draw();

	glPopMatrix();

	float planeExtent = 1.2;
	float detOffset = 0.3;
	float srcOffset = 3.0;

	glEnable(GL_BLEND);
	glDisable(GL_LIGHTING);
	glDisable(GL_CULL_FACE);
	glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA ); 
	glColor4f(1.0,1.0,1.0,0.4);


	//Draw the slicing plane
	glBegin(GL_TRIANGLES);
	glVertex3f(0,1.5*detOffset,planeExtent);
	glVertex3f(0,1.5*detOffset,-planeExtent);
	glVertex3f(0,-detOffset,planeExtent);

	glVertex3f(0,1.5*detOffset,-planeExtent);
	glVertex3f(0,-detOffset,-planeExtent);
	glVertex3f(0,-detOffset,planeExtent);
	glEnd();

	float detExtent = 1.2;


	glColor4f(1.0,1.0,1.0,0.2);

	//First, draw the detector
	glBegin(GL_QUADS);
	glVertex3f(detExtent,-detOffset,detExtent);
	glVertex3f(-detExtent,-detOffset,detExtent);
	glVertex3f(-detExtent,-detOffset,-detExtent);
	glVertex3f(detExtent,-detOffset,-detExtent);
	glEnd();

	//Then, draw rays joining the source to the ends of the detector
	glColor4f(1.0,1.0,1.0,0.5f);
	glBegin(GL_LINES);
	glVertex3f(0,srcOffset,0);
	glVertex3f(detExtent,-detOffset,detExtent);
	glVertex3f(0,srcOffset,0);
	glVertex3f(-detExtent,-detOffset,detExtent);
	glVertex3f(0,srcOffset,0);
	glVertex3f(-detExtent,-detOffset,-detExtent);
	glVertex3f(0,srcOffset,0);
	glVertex3f(detExtent,-detOffset,-detExtent);
	glEnd();

	glColor3f(1.0,0.0,0.0);


	glDisable(GL_LIGHTING);


}

void GLWidget::resizeGL(int width, int height)
{
	if(height == 0)
		height = 1;
	glViewport(0, 0, (GLint)width, (GLint)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLfloat)width/(GLfloat)height, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	switch(event->key())
	{
	case Qt::Key_F2:break;

	}
}
void GLWidget::mousePressEvent(QMouseEvent *event)
{
	lastPos = event->pos();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	int dx = event->x() - lastPos.x();
	int dy = event->y() - lastPos.y();

	if (event->buttons() & Qt::LeftButton) {
		setXRotation(xRot + 8 * dy);
		setYRotation(yRot + 8 * dx);
	} else if (event->buttons() & Qt::RightButton) {
		setXRotation(xRot + 8 * dy);
		setZRotation(zRot + 8 * dx);
	}
	lastPos = event->pos();
}

void GLWidget::wheelEvent(QWheelEvent *e)
{
	float deltaZ = 0.2;
	e->delta() > 0 ? zoom += deltaZ : zoom -= deltaZ;
	if(zoom<=0.6)
		deltaZ = 0.05;
	else
		deltaZ = 0.2;
	if(zoom<0.1)
		zoom=0.1;
	//updateGL();
}
