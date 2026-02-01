import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class AnalysisResult(db.Model):
    """Store uploaded file analysis results."""
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    total_users = db.Column(db.Integer, nullable=False)
    risk_flagged = db.Column(db.Integer, nullable=False)
    avg_risk_score = db.Column(db.Float, nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<AnalysisResult {self.filename}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'total_users': self.total_users,
            'risk_flagged': self.risk_flagged,
            'avg_risk_score': round(self.avg_risk_score, 3),
            'created_at': self.created_at.isoformat(),
        }
