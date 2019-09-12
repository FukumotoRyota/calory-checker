import MySQLdb

class MySQLCon():
  def __init__(self, host, port, user, passwd, db):
    self.host = host
    self.port = port
    self.user = user
    self.passwd = passwd
    self.db = db

  def query(self, sql):
    connect = MySQLdb.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db)
    cursor = connect.cursor()
    cursor.execute(sql)
    for row in cursor:
      print(row[0], row[1])
    cursor.close()
    connect.close()