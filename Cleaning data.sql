use BOL;

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- DATA CLEANING: Removing rows (incl. counts) : 6,516 orders are thrown out
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Count is 4,779,466
select count(*) [count] 
from dbo.data_imported

-- Non-sensical orders (there is some overlap)
select sum(case when orderDate > cancellationDate												then 1 else 0 end) as [01. Cancelled before ordered],					-- occurs    0 times
       sum(case when orderDate > promisedDeliveryDate				 							then 1 else 0 end) as [02. Delivery promised before ordered],			-- occurs   76 times
       sum(case when orderDate > shipmentDate													then 1 else 0 end) as [03. Shipped before ordered],						-- occurs    0 times
       sum(case when orderDate > convert(date, dateTimeFirstDeliveryMoment)						then 1 else 0 end) as [04. Delivered before ordered],					-- occurs   34 times
       sum(case when orderDate > returnDate														then 1 else 0 end) as [05. Returned before ordered],					-- occurs    0 times 
       sum(case when orderDate > startDateCase													then 1 else 0 end) as [06. Case opened before ordered],					-- occurs  170 times
       sum(case when orderDate < registrationDateSeller 										then 1 else 0 end) as [07. Seller did not exist yet],					-- occurs    0 times
       sum(case when cancellationDate > convert(date, dateTimeFirstDeliveryMoment)				then 1 else 0 end) as [08. Cancelled after delivery],					-- occurs   19 times
       sum(case when cancellationDate > returnDate												then 1 else 0 end) as [09. Cancelled after returned],					-- occurs    1 time
       sum(case when cancellationDate > shipmentDate and cancellationReasonCode like '%CUST%'	then 1 else 0 end) as [10. Customer cancelled order after shipping],	-- occurs    3 times
       sum(case when shipmentDate > convert(date, dateTimeFirstDeliveryMoment)					then 1 else 0 end) as [11. Shipped after delivery],						-- occurs 3925 times
       sum(case when shipmentDate > returnDate  												then 1 else 0 end) as [12. Shipped after returned],						-- occurs    0 times
       sum(case when returnDate < convert(date, dateTimeFirstDeliveryMoment)					then 1 else 0 end) as [13. Returned before delivery],					-- occurs 1152 times
       sum(case when registrationDateSeller is null												then 1 else 0 end) as [14. Seller is not registered],					-- occurs  234 times
	   sum(case when promisedDeliveryDate is null												then 1 else 0 end) as [15. Promised delivery date unknown],				-- occurs  941 times
       count(*) [count]
from dbo.data_imported

-- Delete all noise (non-sensical rows). Total = 6,516 orders (0.14% of total orders)
delete from dbo.data_cleaned
where 1 = 1
      and
      (
          orderDate > cancellationDate
          or orderDate > promisedDeliveryDate
          or orderDate > shipmentDate
          or orderDate > convert(date, dateTimeFirstDeliveryMoment)
          or orderDate > returnDate
          or orderDate > startDateCase
          or orderDate < registrationDateSeller
          or cancellationDate > convert(date, dateTimeFirstDeliveryMoment)
          or cancellationDate > returnDate
          or (cancellationDate > shipmentDate and cancellationReasonCode like '%CUST%')
          or shipmentDate > convert(date, dateTimeFirstDeliveryMoment)
          or shipmentDate > returnDate
          or returnDate < convert(date, dateTimeFirstDeliveryMoment)
          or registrationDateSeller is null
		  or promisedDeliveryDate is null
      );

-- Count is 4,772,950
select count(*) [count]
from dbo.data_cleaned;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- DATA CLEANING: Cleaning up certain values
-- We assume the determinants are correct. We add adjusted variables to make sure the drivers of determinants perfectly correspond to the match determinants.
-- Cancellation date and reason			   --drive--> noCancellation
-- DateTimeFirstDeliveryMoment			   --drive--> onTimeDelivery
-- Return date, quantity, and reason	   --drive--> noReturn
-- Case date and count of distinct cases   --drive--> noCase

-- Other inconsistency that is fixed with an adjusted variable
-- If quantity returned > quantity ordered --> adjusted quantity returned = quantity ordered

-- NOTE: some cancellations seem to be incorrectly classified. See data validation. Again, we assume that all determinant classifications are correct and hence follow that
-- logic. 
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
alter table dbo.data_cleaned
add adjCancellationDate date null,
    adjCancellationReasonCode nvarchar(max) null,
    adjDateTimeFirstDeliveryMoment datetime2 null,
    adjReturnDate date null,
    adjQuantityReturned int null,
    adjReturnCode varchar(max) null,
    adjStartDateCase date null,
    adjCntDistinctCaseIds int null;

update dbo.data_cleaned
set adjCancellationDate			   = cancellationDate,
    adjCancellationReasonCode	   = cancellationReasonCode,
    adjDateTimeFirstDeliveryMoment = dateTimeFirstDeliveryMoment,
    adjReturnDate				   = returnDate,
    adjQuantityReturned			   = quantityReturned,
    adjReturnCode				   = returnCode,
    adjStartDateCase			   = startDateCase,
    adjCntDistinctCaseIds		   = cntDistinctCaseIds;

update dbo.data_cleaned
set adjCancellationDate = null,
    adjCancellationReasonCode = null
where noCancellation = 1;

update dbo.data_cleaned
set adjDateTimeFirstDeliveryMoment = null
where onTimeDelivery is null;

update dbo.data_cleaned
set adjReturnDate = null,
    adjQuantityReturned = null,
    adjReturnCode = null
where noReturn = 1;

update dbo.data_cleaned
set adjQuantityReturned = quantityOrdered
where quantityReturned > quantityOrdered;

update dbo.data_cleaned
set adjStartDateCase = null,
    adjCntDistinctCaseIds = null
where noCase = 1;

-----------------------------------------------------------------------------------------------
-- CHECK
-----------------------------------------------------------------------------------------------
-- Cancellations
select *
from dbo.data_cleaned
where (adjCancellationDate is not null and noCancellation = 1)
      or (adjCancellationDate is null and noCancellation = 0);

-- Deliveries
select *
from dbo.data_cleaned
where (   --- Should be on time
          convert(date, adjDateTimeFirstDeliveryMoment) <= promisedDeliveryDate
          and (onTimeDelivery = 0 or onTimeDelivery is null)
      )
      or
      (   --- Should be late
          convert(date, adjDateTimeFirstDeliveryMoment) > promisedDeliveryDate
          and (onTimeDelivery = 1 or onTimeDelivery is null)
      )   --- Should be unknown
      or (adjDateTimeFirstDeliveryMoment is null and onTimeDelivery in ( 0, 1 ));

-- Returns
select *
from dbo.data_cleaned
where (adjReturnDate is not null and noReturn = 1)
      or (adjReturnDate is null and noReturn = 0);

-- Cases
select *
from dbo.data_cleaned
where (adjStartDateCase is not null and noCase = 1)
      or (adjStartDateCase is null and noCase = 0);

-- Return quantity inconsistent 
select *
from dbo.data_cleaned
where adjQuantityReturned > quantityOrdered;

-------------------------------------------------------------------------------------------------------------------------------------
-- INSIGHTS
-------------------------------------------------------------------------------------------------------------------------------------
select count(*) [0. Total amount],
       sum(iif(cancellationDate is not null            and adjCancellationDate is null,            1, 0)) [1. Cancellations],
       sum(iif(dateTimeFirstDeliveryMoment is not null and adjDateTimeFirstDeliveryMoment is null, 1, 0)) [2. Deliveries],
       sum(iif(returnDate is not null                  and adjReturnDate is null,                  1, 0)) [3. Returns],
       sum(iif(startDateCase is not null               and adjStartDateCase is null,               1, 0)) [4. Cases],
       sum(iif(quantityReturned <> adjQuantityReturned, 1, 0)) [5. Quantity Returned]
from dbo.data_cleaned
where 1 = 1
      and
      ((cancellationDate is not null               and adjCancellationDate is null)
       or (dateTimeFirstDeliveryMoment is not null and adjDateTimeFirstDeliveryMoment is null)
       or (returnDate is not null                  and adjReturnDate is null)
       or (startDateCase is not null               and adjStartDateCase is null)
       or (quantityReturned <> adjQuantityReturned)
      );

-- Including overlap
select case when cancellationDate is not null            and adjCancellationDate is null            then 1 else 0 end as cancellations,
       case when dateTimeFirstDeliveryMoment is not null and adjDateTimeFirstDeliveryMoment is null then 1 else 0 end as deliveries,
       case when returnDate is not null                  and adjReturnDate is null                  then 1 else 0 end as [returns],
       case when startDateCase is not null               and adjStartDateCase is null               then 1 else 0 end as cases,
       case when quantityReturned <> adjQuantityReturned then 1 else 0 end as [quantity returned],
       count(*) [count]
from dbo.data_cleaned
where 1 = 1
      and
      ((cancellationDate is not null               and adjCancellationDate is null)
       or (dateTimeFirstDeliveryMoment is not null and adjDateTimeFirstDeliveryMoment is null)
       or (returnDate is not null                  and adjReturnDate is null)
       or (startDateCase is not null               and adjStartDateCase is null)
       or (quantityReturned <> adjQuantityReturned)
      )
group by case when cancellationDate is not null            and adjCancellationDate is null            then 1 else 0 end,
         case when dateTimeFirstDeliveryMoment is not null and adjDateTimeFirstDeliveryMoment is null then 1 else 0 end,
         case when returnDate is not null                  and adjReturnDate is null                  then 1 else 0 end,
         case when startDateCase is not null               and adjStartDateCase is null				  then 1 else 0 end,
         case when quantityReturned <> adjQuantityReturned then 1 else 0 end
order by [count] desc;