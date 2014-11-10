#include "window.h"

#include <QVBoxLayout>
#include <QGroupBox>
#include <QTimer>

Window::Window()
{

	glwidget = new GLWidget;
	xraywidget = new XRayWidget;

	glwidget->setFixedSize(QSize(512,512));
	xraywidget->setFixedSize(QSize(512,512));

	QHBoxLayout *layout = new QHBoxLayout;
	layout->addWidget(xraywidget);
	layout->addWidget(glwidget);

	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(repaintGL()));
	timer->start(2);



	setLayout(layout);
	setWindowTitle(tr("X-Ray Simulation"));
}

void Window::repaintGL()
{
	glwidget->repaint();
	xraywidget->repaint();
}
