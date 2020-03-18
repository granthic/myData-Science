import sys

sys.path.insert(1, r'C:\Users\pmarathe\Documents\Healthfirst\Project\AutoQA\SalesforceDataSync')
from commonUtil.APIClient import APIHelperClass
from commonUtil.dbHelper import DBHelperClass


class UtilHelperClass:

    def __init__(self):
        j = 0
        mysql_conn=""
        my_DBHelper = DBHelperClass()
        mysql_conn = my_DBHelper.create_mySQLConnection(dbFilePath)

    def insert_row_with_param(self,conn,qSelect,params):

        dbFilePath = dbFilePath = r"C:\Users\pmarathe\Documents\Healthfirst\Project\AutoQA\SalesforceDataSync\SF_ODS_Sync.db"
        insert_sql="INSERT INTO notes_taken (topic,notes,comments) VALUES (?,?,?)"
    #insert_sql="INSERT INTO notes_taken (topic,notes,comments) VALUES ('topic','notes','comments')"
        row_id= my_DBHelper.insert_row_with_param(mysql_conn,insert_sql,params)
        return row_id