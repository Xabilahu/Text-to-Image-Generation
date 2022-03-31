import datetime
import os
import signal
import sqlite3
import sys
import time

from constants import DATABASE_INIT_SCRIPT, DATABASE_PATH


class DB(object):
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = object.__new__(cls)

        return cls._singleton

    def __init__(cls):
        if not os.path.exists(DATABASE_PATH):
            with open(DATABASE_INIT_SCRIPT, "r") as f:
                contents = f.read()

            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.executescript(contents)
            conn.commit()
            cursor.close()
            conn.close()

        cls.conn = sqlite3.connect(DATABASE_PATH)

    def close(self):
        self.conn.close()
        DB._singleton = None

    def _query_one(self, *args):
        if len(args) > 2:
            raise RuntimeError(
                "_query_one expects a maximum of 2 arguments: (query: str, [params: tuple])"
            )

        cursor = self.conn.cursor()
        cursor.execute(*args)
        row = cursor.fetchone()
        cursor.close()
        return row

    def _update(self, *args):
        lastrowid = None

        try:
            cursor = self.conn.cursor()
            cursor.execute(*args)
            self.conn.commit()
            lastrowid = cursor.lastrowid
        except sqlite3.Error:
            print(f"[ERROR] Could not perform the operation: {args}", file=sys.stderr)
            self.conn.rollback()
        finally:
            cursor.close()

        return lastrowid

    def insert_job(self, image_config):
        image_id = self._update(
            "INSERT INTO images (prompt, dirname, model_name, seed, step_size, update_freq, optimization_steps, similarity_factor, smoothing_factor, truncation_factor, imagenet_classes, optimize_class) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_config["text"],
                image_config["dirname"],
                image_config["model_name"],
                image_config["seed"],
                image_config["step_size"],
                image_config["update_freq"],
                image_config["optimization_steps"],
                image_config["similarity_factor"],
                image_config["smoothing_factor"],
                image_config["truncation_factor"],
                image_config["imagenet_classes"],
                image_config["optimize_class"],
            ),
        )

        self._update(
            "INSERT INTO job_queue (process_id, image_id) VALUES (?, ?)",
            (os.getpid(), image_id),
        )

        return image_id

    def insert_vote(self, vote_type, image_id, session_id, vote_score):
        query_result = self._query_one(
            "SELECT * FROM votes WHERE session_id = ? AND image_id = ?",
            (
                session_id,
                image_id,
            ),
        )

        if query_result is None:
            self._update(
                f"INSERT INTO votes (session_id, image_id, {vote_type}) VALUES (?, ?, ?)",
                (session_id, image_id, vote_score),
            )
        else:
            self._update(
                f"UPDATE votes SET {vote_type} = ? WHERE session_id = ? AND image_id = ?",
                (vote_score, session_id, image_id),
            )

    def update_image(self, image_id, gif_filename, image_filename):
        self._update(
            "UPDATE images SET gif_filename = ?, image_filename = ? WHERE id = ?",
            (gif_filename, image_filename, image_id),
        )

    def update_start_time(self, job_id):
        self._update(
            "UPDATE job_queue SET start_time = ?, status = 'PROCESSING' WHERE id = ?",
            (datetime.datetime.now(), job_id),
        )

    def update_end_time(self, job_id):
        self._update(
            "UPDATE job_queue SET end_time = ?, status = 'FINISHED' WHERE id = ?",
            (datetime.datetime.now(), job_id),
        )

    def query_image(self, image_id):
        return self._query_one("SELECT * FROM images WHERE id = ?", (image_id,))

    def query_image_id(self, dirname, filename, colname):
        return self._query_one(
            f"SELECT id FROM images WHERE dirname = ? AND {colname} = ?",
            (
                dirname,
                filename,
            ),
        )

    def query_image_filenames(self, image_id):
        return self._query_one(
            "SELECT dirname, gif_filename, image_filename FROM images WHERE id = ?",
            (image_id,),
        )

    def query_pending_jobs(self):
        return self._query_one(
            "SELECT id, process_id, image_id FROM job_queue WHERE status == 'PENDING' ORDER BY insert_time ASC"
        )

    def get_random_non_voted_image_id(self, session_id):
        return self._query_one(
            "SELECT id FROM images WHERE id NOT IN (SELECT image_id FROM votes WHERE session_id = ?) ORDER BY RANDOM()",
            (session_id,),
        )


def poll_pending_jobs(poll_freq=1):
    while True:
        db = DB()
        job = db.query_pending_jobs()

        if job is not None:
            db.update_start_time(job[0])
            db.close()
            return job

        time.sleep(poll_freq)


def get_image_config(image_id):
    db = DB()
    image_data = db.query_image(image_id)
    config = {
        "text": list(map(str.strip, image_data[1].split("|"))),
        "dirname": image_data[2],
        "model_name": image_data[5],
        "seed": image_data[6],
        "step_size": image_data[7],
        "update_freq": image_data[8],
        "optimization_steps": image_data[9],
        "similarity_factor": image_data[10],
        "smoothing_factor": image_data[11],
        "truncation_factor": image_data[12],
        "imagenet_classes": image_data[13],
        "optimize_class": image_data[14] == 1,
    }
    db.close()
    return config


def update_image(job_id, image_id, gif_filename, image_filename):
    db = DB()
    db.update_image(image_id, gif_filename, image_filename)
    db.update_end_time(job_id)
    db.close()


def dummy_handler(signum, stack_frame):
    pass


def push_job(image_config, image_filenames):
    db = DB()
    image_id = db.insert_job(image_config)
    db.close()

    # Wait for job_executor to process the inserted job
    # TODO: add signal.SIGUSR2 to update message with position in queue
    signal.signal(signal.SIGUSR1, dummy_handler)
    signal.pause()

    db = DB()
    dirname, gif_filename, img_filename = db.query_image_filenames(image_id)
    image_filenames[0] = (
        os.path.join(dirname, gif_filename)
        if gif_filename != "" and gif_filename is not None
        else None
    )
    image_filenames[1] = (
        os.path.join(dirname, img_filename)
        if img_filename != "" and img_filename is not None
        else None
    )
    db.close()


def get_random_image_id(session_id):
    db = DB()
    image_id = db.get_random_non_voted_image_id(session_id)
    db.close()
    return image_id[0]


def get_image_data(image_id):
    db = DB()
    data = db.query_image(image_id)
    db.close()
    gif_filename = (
        os.path.join(data[2], data[3])
        if data[3] != "" and data[3] is not None
        else None
    )
    image_filename = (
        os.path.join(data[2], data[4])
        if data[4] != "" and data[4] is not None
        else None
    )
    return data[1], gif_filename, image_filename


def process_vote(vote_type, vote_score, image_id, session_id):
    db = DB()

    if vote_type not in ["quality", "agreement"]:
        raise RuntimeError(f"Unrecognized vote_type: {vote_type}")

    db.insert_vote(vote_type, image_id, session_id, vote_score)
    db.close()


def image_id_from_filename(filename):
    db = DB()

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    if os.path.splitext(filename)[1] == ".gif":
        db_col = "gif_filename"
    else:
        db_col = "image_filename"

    image_id = db.query_image_id(dirname, filename, db_col)
    db.close()
    return image_id[0]
