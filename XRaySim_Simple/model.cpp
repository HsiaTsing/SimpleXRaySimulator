
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "cutil_math.h"
#include "GL/glew.h"
#include <math.h>


void Model::draw()
{

	glBindBuffer(GL_ARRAY_BUFFER,m_posVBO);
	glVertexPointer(3,GL_FLOAT,0,0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, m_normVBO);
	glNormalPointer(GL_FLOAT, 0, 0);
	glEnableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_elemVBO);

	glDrawElements(GL_TRIANGLES,m_nFaces*3,GL_UNSIGNED_INT,0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
};
void Model::init()
{

	glewInit();

	glGenBuffers(1,&m_posVBO);
	glGenBuffers(1,&m_normVBO);
	glGenBuffers(1,&m_elemVBO);

	glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*m_nPoints , m_hPoints, GL_DYNAMIC_DRAW); 

	glBindBuffer(GL_ARRAY_BUFFER, m_normVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*m_nNormals , m_hNormals, GL_DYNAMIC_DRAW); 

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint3)*m_nFaces , m_hFaces ,GL_STATIC_DRAW);

	
}

void Model::load(const char* filename, float SF, bool recenter, bool recalcNormals)
{

	FILE* fpin = fopen(filename,"r");

	if(fpin == NULL)
		return;

	scaleFactor = SF;

	m_nPoints=0;
	m_nFaces=0;
	m_nNormals=0; 

	char str[100];

	while(fgets(str,100,fpin)!=NULL)
	{

		if((str[0] == 'v') && (str[1]==' '))
			m_nPoints++;

		else if(str[0]=='f')
			m_nFaces++;

		else if((str[0]=='v') && (str[1]=='n'))
			m_nNormals++;

	}
	
	if(recalcNormals)
		m_nNormals = m_nPoints;

	m_hPoints = (float3*)malloc(sizeof(float3)*m_nPoints);
	m_hFaces = (uint3*)malloc(sizeof(uint3)*m_nFaces);
	m_hNormals = (float3*)malloc(sizeof(float3)*m_nNormals);

	rewind(fpin);

	int pointsIndex = 0;
	int facesIndex = 0;
	int normalsIndex = 0;


	m_min = make_float3(100.0);
	m_max = make_float3(-100.0);;

	while(fgets(str,100,fpin)!=NULL)
	{
		if(str[0] == 'v' && str[1] == ' ')
		{
			float tmp1,tmp2,tmp3;
			sscanf(&str[1]," %f %f %f",&tmp1,&tmp2,&tmp3);

			m_min.x = (tmp1<m_min.x)?tmp1:m_min.x;
			m_min.y = (tmp2<m_min.y)?tmp2:m_min.y;
			m_min.z = (tmp3<m_min.z)?tmp3:m_min.z;

			m_max.x = (tmp1>m_max.x)?tmp1:m_max.x;
			m_max.y = (tmp2>m_max.y)?tmp2:m_max.y;
			m_max.z = (tmp3>m_max.z)?tmp3:m_max.z;

			m_hPoints[pointsIndex].x = tmp1;
			m_hPoints[pointsIndex].y = tmp2;
			m_hPoints[pointsIndex].z = tmp3;
			pointsIndex++;
		}
		else if(str[0] == 'v' && str[1] == 'n')
		{
			if(!recalcNormals)
			{
				float tmp1,tmp2,tmp3;
				sscanf(&str[2]," %f %f %f",&tmp1,&tmp2,&tmp3);

				m_hNormals[normalsIndex].x = tmp1;
				m_hNormals[normalsIndex].y = tmp2;
				m_hNormals[normalsIndex].z = tmp3;
				normalsIndex++;
			}
		}
		else if(str[0] == 'f')
		{
			int tmp1,tmp2,tmp3;
			int tmp;

			sscanf(&str[1]," %d\/%d\/%d %d\/%d\/%d %d\/%d\/%d",&tmp1,&tmp,&tmp,&tmp2,&tmp,&tmp,&tmp3,&tmp,&tmp);
			m_hFaces[facesIndex].x = tmp1-1;
			m_hFaces[facesIndex].y = tmp2-1;
			m_hFaces[facesIndex].z = tmp3-1;
			facesIndex++;
		}

	}

	fclose(fpin);

	if(recalcNormals){
		float3 temp;

		float e1x=0,e1y=0,e1z=0;
		float e2x=0,e2y=0,e2z=0;
		float nx,ny,nz;

		for(int i = 0; i< m_nFaces; i++)
		{
			e1x = m_hPoints[m_hFaces[i].y].x - m_hPoints[m_hFaces[i].x].x;
			e1y = m_hPoints[m_hFaces[i].y].y - m_hPoints[m_hFaces[i].x].y;
			e1z = m_hPoints[m_hFaces[i].y].z - m_hPoints[m_hFaces[i].x].z;

			e2x = m_hPoints[m_hFaces[i].z].x - m_hPoints[m_hFaces[i].x].x;
			e2y = m_hPoints[m_hFaces[i].z].y - m_hPoints[m_hFaces[i].x].y;
			e2z = m_hPoints[m_hFaces[i].z].z - m_hPoints[m_hFaces[i].x].z;

			nx = e2z*e1y - e1z*e2y;
			ny = e1z*e2x - e2z*e1x;
			nz = e1x*e2y - e2x*e1y;

			float mag = nx*nx+ny*ny+nz*nz;
			mag = sqrt(mag);

			m_hNormals[m_hFaces[i].x].x = nx/mag;
			m_hNormals[m_hFaces[i].x].y = ny/mag;
			m_hNormals[m_hFaces[i].x].z = nz/mag;

			m_hNormals[m_hFaces[i].y].x = nx/mag;
			m_hNormals[m_hFaces[i].y].y = ny/mag;
			m_hNormals[m_hFaces[i].y].z = nz/mag;

			m_hNormals[m_hFaces[i].z].x = nx/mag;
			m_hNormals[m_hFaces[i].z].y = ny/mag;
			m_hNormals[m_hFaces[i].z].z = nz/mag;
		}
	}

	computeBoundingBox(recenter);

}

void Model::computeBoundingBox(bool recenter)
{
	if(recenter)
	{

		float3 center = {0,0,0};


		for(int i = 0; i < m_nPoints; i++)
		{
			center.x += m_hPoints[i].x;
			center.y += m_hPoints[i].y;
			center.z += m_hPoints[i].z;
		}

		center.x /= m_nPoints;
		center.y /= m_nPoints;
		center.z /= m_nPoints;

		for(int i = 0; i < m_nPoints; i++)
		{
			m_hPoints[i].x -= center.x;
			m_hPoints[i].y -= center.y;
			m_hPoints[i].z -= center.z;
		}
	}


	float sem1 = abs(m_max.x) + abs(m_min.x);
	float sem2 = abs(m_max.y) + abs(m_min.y);
	float sem3 = abs(m_max.z) + abs(m_min.z);

	float maxVal = sem1;

	if(sem2>maxVal)
		maxVal = sem2;
	if(sem3>maxVal)
		maxVal = sem3;

	// @@@@ Put the scale variation in the following line!!
	maxVal /= scaleFactor;

	for(int i = 0; i < m_nPoints; i++)
	{
		m_hPoints[i].x *= 1/maxVal;
		m_hPoints[i].y *= 1/maxVal;
		m_hPoints[i].z *= 1/maxVal;
	}
}