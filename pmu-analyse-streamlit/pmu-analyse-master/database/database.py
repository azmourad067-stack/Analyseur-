import logging
from datetime import datetime

from sqlalchemy.orm import sessionmaker
from database.setup_database import engine, Hippodrome, Pays, Reunion, Course, Participant


def _session():
    return sessionmaker(bind=engine)()


def save_pays(pays_data):
    if not pays_data:
        return
    session = _session()
    try:
        existing = session.query(Pays).filter_by(code=pays_data.get('code')).first()
        if not existing:
            session.add(Pays(**pays_data))
            logging.info("Saving pays data")
        session.commit()
    finally:
        session.close()


def save_hippodrome(hippodrome_data):
    if not hippodrome_data:
        return
    session = _session()
    try:
        existing = session.query(Hippodrome).filter_by(code=hippodrome_data.get('code')).first()
        if not existing:
            session.add(Hippodrome(**hippodrome_data))
            logging.info("Saving hippodrome data")
        session.commit()
    finally:
        session.close()


def save_reunions(reunion_data):
    session = _session()
    try:
        existing = (session.query(Reunion)
                    .filter_by(dateReunion=reunion_data.get('dateReunion'),
                               numOfficiel=reunion_data.get('numOfficiel'))
                    .first())
        if not existing:
            # Copie pour ne pas muter le dictionnaire d'origine
            data = dict(reunion_data)
            hippodrome_code = (data.get('hippodrome') or {}).get('code')
            pays_code = (data.get('pays') or {}).get('code')
            data.pop('hippodrome', None)
            data.pop('pays', None)
            data.pop('courses', None)
            session.add(Reunion(**data, hippodrome_code=hippodrome_code, pays_code=pays_code))
            logging.info("Saving reunion data")
        session.commit()
    finally:
        session.close()


def save_courses(courses_data):
    session = _session()
    try:
        for course_data in courses_data:
            course_data['heureDepart'] = datetime.utcfromtimestamp(course_data['heureDepart'] / 1000.0)
            existing = (session.query(Course).filter_by(
                heureDepart=course_data.get('heureDepart'),
                numReunion=course_data.get('numReunion'),
                numOrdre=course_data.get('numOrdre')
            ).first())
            if not existing:
                hippodrome_code = (course_data.get('hippodrome') or {}).get('codeHippodrome')
                valid_attributes = [attr.name for attr in Course.__table__.columns]
                filtered_course_data = {k: v for k, v in course_data.items() if k in valid_attributes}
                session.add(Course(**filtered_course_data, hippodrome_code=hippodrome_code))
                logging.info("Saving Course data")
        session.commit()
    finally:
        session.close()


def save_participants(participants_data, course_data, reunion_data):
    session = _session()
    try:
        for p in participants_data:
            existing = (session.query(Participant)
                        .filter_by(idCheval=p.get('idCheval'),
                                   numReunion=course_data.get('numReunion'),
                                   numOrdre=course_data.get('numOrdre'))
                        .first())
            if existing:
                continue
            gp = p.get('gainsParticipant') or {}
            session.add(Participant(
                idCheval=p.get('idCheval'),
                numPmu=p.get('numPmu'),
                nom=p.get('nom'),
                age=p.get('age'),
                sexe=p.get('sexe'),
                race=p.get('race'),
                statut=p.get('statut'),
                placeCorde=p.get('placeCorde'),
                oeilleres=p.get('oeilleres'),
                proprietaire=p.get('proprietaire'),
                entraineur=p.get('entraineur'),
                driver=p.get('driver'),
                musique=p.get('musique'),
                nombreCourses=p.get('nombreCourses'),
                nombreVictoires=p.get('nombreVictoires'),
                nombrePlaces=p.get('nombrePlaces'),
                gainsCarriere=gp.get('gainsCarriere'),
                gainsVictoires=gp.get('gainsVictoires'),
                gainsPlace=gp.get('gainsPlace'),
                gainsAnneeEnCours=gp.get('gainsAnneeEnCours'),
                dateReunion=reunion_data.get('dateReunion'),
                numReunion=course_data.get('numReunion'),
                numOrdre=course_data.get('numOrdre'),
                hippodrome_code=(course_data.get('hippodrome') or {}).get('codeHippodrome'),
            ))
        session.commit()
    finally:
        session.close()
