WITH fraud_chargebacks AS (
  SELECT cb.when_created when_created, cb.tx_id tx_id
  FROM chargebacks cb
  LEFT JOIN chargehound_disputes disp USING(tx_id)
  WHERE disp.when_created IS NULL OR disp.state != 'won'
),

ups_blocks AS (
  SELECT when_created, u_id
  FROM user_blocks
  WHERE status = 'FRAUD'
)

SELECT tx.when_created when_created,
       CONCAT(u.send_country, '-', u.receive_country) corridor,
       CASE WHEN cb.when_created IS NULL AND blk.when_created IS NULL
            THEN 0
            ELSE 1
       END AS fraud,
       ip.risk_score
FROM transactions tx
LEFT JOIN fraud_chargebacks cb USING(tx_id)
LEFT JOIN ups_blocks blk USING(u_id)
JOIN users u USING(u_id)
LEFT JOIN ip_addresses_used ip USING(u_id)
WHERE risk_score IS NOT NULL


