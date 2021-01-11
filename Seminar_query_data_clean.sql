-- Remove duplicate rows from the table -> 1.507.270 rows
SELECT DISTINCT * 
INTO #temp_bol_data
FROM bol_data;

-- Count nr. of rows in total data (non-duplicates): 1.507.270 rows
SELECT *
FROM #temp_bol_data;

-- CREATE CLEANED TABLE
-- First drop existing version of the table
DROP TABLE cleaned_bol_data;

-- Then move all rows from the non-duplicates table into the cleaned table
SELECT *
INTO cleaned_bol_data -- cleaned table
FROM #temp_bol_data;

-- Delete all noise (non-sensible rows)
DELETE FROM cleaned_bol_data
WHERE ((startDateCase < orderDate) OR (cancellationDate < orderDate) OR (promisedDeliveryDate < orderDate)
		OR (shipmentDate < orderDate) OR (datetTimeFirstDeliveryMoment < orderDate) OR (returnDateTime < orderDate)
		OR (orderDate < registrationDateSeller) OR (cancellationDate > datetTimeFirstDeliveryMoment) 
		OR (cancellationDate > returnDateTime) OR (CONVERT(date,returnDateTime) < (CONVERT(date,datetTimeFirstDeliveryMoment)) 
		AND datetTimeFirstDeliveryMoment IS NOT NULL AND returnDateTime IS NOT NULL) OR (shipmentDate > returnDateTime)
		OR (shipmentDate > CONVERT(date,datetTimeFirstDeliveryMoment)) OR (registrationDateSeller IS NULL)
		OR (cancellationDate > shipmentDate AND (cancellationReasonCode = 'CUST_FE' OR cancellationReasonCode = 'CUST_CS')));

-- Count the rows in the resulting, cleaned table
SELECT COUNT(*) FROM cleaned_bol_data;

-- CHECKS
-- Check: boolean to classification correct
SELECT onTimeDelivery, noReturn, noCancellation, noCase, detailedMatchClassification, count(*)
FROM #temp_bol_data
WHERE onTimeDelivery IS NULL
GROUP BY onTimeDelivery, noReturn, noCancellation, noCase, detailedMatchClassification;

-- Check whether the boolean variables are constructed correctly according to the related variables (CANCELLATIONS INCORRECT)
WITH diff_return_check AS (
SELECT cleaned_bol_data.*,
		(CASE WHEN (returnDateTime IS NOT NULL AND DATEDIFF(day,orderDate,returnDateTime) <= 29) THEN 0
			ELSE 1 END) - noReturn AS diffReturnCheck,
		(CASE WHEN (startDateCase IS NOT NULL AND DATEDIFF(day,orderDate,startDateCase) <= 29) THEN 0
			ELSE 1 END) - noCase AS diffCaseCheck,
		(CASE WHEN (cancellationDate IS NOT NULL AND DATEDIFF(day,orderDate,cancellationDate) <= 10) THEN 0
			ELSE 1 END) - noCancellation AS diffCancellationCheck,
		CASE WHEN (CASE WHEN (datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) <= 0
					AND datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) < 13) THEN 'True'
			  WHEN (datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) > 0 
					AND datediff(day,promisedDeliveryDate,datetTimeFirstDeliveryMoment) < 13) THEN 'False'
			  ELSE 'NULL' END) <> onTimeDelivery THEN 1 ELSE 0 END AS diffOnTimeDeliveryCheck
FROM cleaned_bol_data)
SELECT diffReturnCheck, diffCaseCheck, diffCancellationCheck, diffOnTimeDeliveryCheck, count(*)
FROM diff_return_check
GROUP BY diffReturnCheck, diffCaseCheck, diffCancellationCheck, diffOnTimeDeliveryCheck;

-- All entries with a cancellation reason for which the noCancellation boolean is 1 (20.721 rows)
SELECT *
FROM cleaned_bol_data
WHERE cancellationReasonCode LIKE 'CUST%' AND noCancellation = 1;


