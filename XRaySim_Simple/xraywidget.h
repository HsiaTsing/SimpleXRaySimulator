#ifndef XRAYGLWIDGET_H
#define XRAYGLWIDGET_H

#include "QGLWidget"
#include "common.h"

class XRayWidget : public QGLWidget
{
	Q_OBJECT

public:
	XRayWidget (QWidget *parent = 0);
	~XRayWidget ();

	GLuint XRayBufferID;
	GLuint XRayTexID;

	QSize minimumSizeHint() const;
	QSize sizeHint() const;

	void xraySim_CUDA();
	rasterData* mRasterData;
protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);
	void keyPressEvent(QKeyEvent *event);

};

#endif