# -*- coding: utf-8 -*-
"""只读守卫测试：证明四层防护挡得住写操作，且限定了可读对象。"""
import sqlite3

import pytest

from tools.db import (DB_PATH, DEFAULT_LIMIT, SqlGuardError, _authorizer, check_sql,
                      query_db)


class TestGuardRejectsWrites:
    @pytest.mark.parametrize("sql", [
        "INSERT INTO alarm_records (turbine_id) VALUES ('T99')",
        "UPDATE maintenance_records SET priority='EMERGENCY'",
        "DELETE FROM alarm_records",
        "DROP TABLE alarm_records",
        "ALTER TABLE alarm_records ADD COLUMN x TEXT",
        "CREATE TABLE evil (id INT)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "PRAGMA query_only = OFF",
        "VACUUM",
    ])
    def test_write_statements_rejected(self, sql):
        result = query_db(sql)
        assert result["ok"] is False
        assert "error" in result

    def test_multi_statement_rejected(self):
        result = query_db("SELECT 1; DELETE FROM alarm_records")
        assert result["ok"] is False
        assert "单条语句" in result["error"]

    def test_comment_hidden_keyword_rejected(self):
        """把 DELETE 藏在注释后换行，仍应被拦。"""
        result = query_db("SELECT 1 /* x */ ; DELETE FROM alarm_records")
        assert result["ok"] is False

    def test_engine_layer_blocks_write_even_without_regex(self):
        """绕过 L4 直连时，L1~L3 仍必须拒绝写入。"""
        conn = sqlite3.connect("file:%s?mode=ro" % DB_PATH.as_posix(), uri=True)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM alarm_records")
        conn.close()


class TestGuardAllowsReads:
    def test_plain_select(self):
        result = query_db("SELECT turbine_id FROM alarm_records WHERE turbine_id='T01'")
        assert result["ok"] is True
        assert result["row_count"] > 0

    def test_with_cte_allowed(self):
        result = query_db(
            "WITH t AS (SELECT * FROM alarm_records "
            "WHERE turbine_id='T03' AND fault_code='24002') "
            "SELECT COUNT(*) AS n FROM t"
        )
        assert result["ok"] is True
        assert result["rows"][0]["n"] == 5  # T03/24002 全期共 5 条

    def test_auto_limit_appended(self):
        result = query_db("SELECT * FROM alarm_records")
        assert result["ok"] is True
        assert "LIMIT %d" % DEFAULT_LIMIT in result["sql"]

    def test_existing_limit_preserved(self):
        assert check_sql("SELECT 1 LIMIT 3").count("LIMIT") == 1

    def test_empty_result_is_not_an_error(self):
        """空结果必须是成功且带说明，不能让模型误判为查询失败而去编造。"""
        result = query_db(
            "SELECT * FROM maintenance_records WHERE turbine_id='T07' AND fault_code='24012'"
        )
        assert result["ok"] is True
        assert result["row_count"] == 0
        assert result["note"] is not None


class TestErrorFeedback:
    def test_bad_column_returns_readable_error(self):
        """SQL 错误要能回灌给模型重写，而不是抛异常中断循环。"""
        result = query_db("SELECT no_such_column FROM alarm_records")
        assert result["ok"] is False
        assert "SQL 执行失败" in result["error"]

    def test_empty_sql(self):
        with pytest.raises(SqlGuardError):
            check_sql("   ")


class TestReadObjectWhitelist:
    """四层防护此前拦的全是**写**，没有一层限制**读什么**。

    对抗性测试实测：`SELECT name, sql FROM sqlite_master` 放行，换个中性问法
    模型就把建表语句原样吐出来——挡住它的是模型的判断，不是工具的边界。
    """

    @pytest.mark.parametrize("sql", [
        "SELECT name, sql FROM sqlite_master",
        "select NAME from SQLITE_MASTER",
        "SELECT 1 FROM alarm_records UNION SELECT length(sql) FROM sqlite_master",
        "SELECT turbine_id FROM alarm_records WHERE turbine_id IN (SELECT name FROM sqlite_master)",
        "SELECT * FROM pragma_table_info('alarm_records')",
    ])
    def test_internal_objects_rejected(self, sql):
        result = query_db(sql)
        assert result["ok"] is False
        assert "内部元数据" in result["error"]

    def test_authorizer_denies_even_when_the_regex_misses(self):
        """L4 正则是最弱的一环，边界由 L3 守：读对象不在白名单一律拒。"""
        assert _authorizer(sqlite3.SQLITE_READ, "sqlite_master", "name") == sqlite3.SQLITE_DENY
        assert _authorizer(sqlite3.SQLITE_READ, "alarm_records", "turbine_id") == sqlite3.SQLITE_OK

    @pytest.mark.parametrize("sql", [
        "SELECT COUNT(*) AS n FROM alarm_records",
        "SELECT turbine_id, COUNT(*) c FROM alarm_records GROUP BY turbine_id",
        "WITH x AS (SELECT * FROM maintenance_records WHERE status='COMPLETED') SELECT COUNT(*) FROM x",
        ("SELECT a.turbine_id, m.work_order_id FROM alarm_records a "
         "JOIN maintenance_records m ON a.turbine_id=m.turbine_id AND a.fault_code=m.fault_code"),
    ])
    def test_business_queries_unaffected(self, sql):
        assert query_db(sql)["ok"] is True
