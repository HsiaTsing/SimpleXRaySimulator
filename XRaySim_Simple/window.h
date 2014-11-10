#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include "glwidget.h"
#include "xraywidget.h"

class Window : public QWidget
{
	Q_OBJECT

public:
	Window();

	GLWidget *glwidget;
	XRayWidget *xraywidget;

private slots:
	void repaintGL();
};

#endif // WINDOW_H
