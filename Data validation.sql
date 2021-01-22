use BOL;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- CHECK CONSTRUCTION OF MATCH DETERMINANTS: UNRECOGNIZED CANCELLATIONS
-- Queries are constructed as follows:
-- The where statements can be split into two parts: 
--		1. The match determinant in question (for instance 'noReturn') is yes (so no return)
--		2. The match determinant in question is no (so a return)
-- For each part, the first (few) boolean(s) are the requirements for the match determinants according to the logic.
-- The last boolean is a contradicting value for the match determinant.
-- HENCE, if the logic of the match determinant classification is perfectly correct, the query should return no rows.
-- If it does, the logic of the classification and the actual classification contradict each other. 

-- The first query is fully explained as an example
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Cancellations. For different type of cancellations, there are varying requirements for a cancellation classification. The types have been split up into seller cancellations,
-- customer cancellations, and other (non-seller & non-customer) cancellations.

-- Consistency of seller cancellations: ALL GOOD
select orderDate,
       cancellationDate,
       promisedDeliveryDate,
       datediff(day, orderDate, cancellationDate) [cancellation time],
       cancellationReasonCode,
       noCancellation,
       *
from dbo.data_cleaned
where cancellationReasonCode like 'SELLER%' --Filter out all orders that are not seller cancellations
      and --- the cancellation should be within 10 days --- contradicting classification
      ((datediff(day, orderDate, cancellationDate) < 11 and noCancellation = 1)
       or --- the cancellation is not within 10 days ---- or there is no cancellation date
       ((datediff(day, orderDate, cancellationDate) >= 11 or cancellationDate is null)
		--- contradicting classification
        and noCancellation = 0
       )
      )
order by 1,
         2;

-- Consistency of customer cancellations: ALL GOOD
select orderDate,
       cancellationDate,
	   promisedDeliveryDate,
       datediff(day, orderDate, cancellationDate) [cancellation time],
	   datediff(day, promisedDeliveryDate, cancellationDate) [time between promised delivery and cancellation],
	   cancellationReasonCode,
       noCancellation,
       *
from dbo.data_cleaned
where cancellationReasonCode like 'CUST%'
          and
          (
              (
                  datediff(day, orderDate, cancellationDate) < 11
                  and cancellationDate > promisedDeliveryDate
                  and noCancellation = 1
              )
              or
              (
                  (
                      datediff(day, orderDate, cancellationDate) >= 11
                      or cancellationDate <= promisedDeliveryDate
                      or cancellationDate is null
                  )
                  and noCancellation = 0
              )
          )
order by 1,
         2;

-- Consistency of non-seller and non-customer cancellations: SOME INCONSISTENCIES
-- NOTE: there seem to be some inconsistencies in the data here. According to our most up-to-date knowledge, these orders should be classified as having a cancellation. 
-- The noCancellation match determinant, however, says otherwise. Our starting point is that the classifications are correct, since we suspect the logic behind the 
-- classifications to be a bit more complicated and there are valid reasons these orders are classified as having no cancellation. Hence, we will treat them as such.
select orderDate,
       cancellationDate,
       promisedDeliveryDate,
       datediff(day, orderDate, cancellationDate) [cancellation time],
       cancellationReasonCode,
       noCancellation,
       *
from dbo.data_cleaned
where (cancellationReasonCode not like 'CUST%' and cancellationReasonCode not like 'SELLER%')
      and
      ((datediff(day, orderDate, cancellationDate) < 11 and noCancellation = 1)
       or
       ((datediff(day, orderDate, cancellationDate) >= 11 or cancellationDate is null)
        and noCancellation = 0
       )
      )
order by 1,
         2;

-- Deliveries: ALL GOOD
select *
from dbo.data_cleaned
where (   -- Should be on time delivery
          datediff(day, orderDate, dateTimeFirstDeliveryMoment) < 13
          and convert(date, dateTimeFirstDeliveryMoment) <= promisedDeliveryDate
          and (onTimeDelivery = 0 or onTimeDelivery is null)
      )
      or
      (   -- Should be late delivery
          datediff(day, orderDate, dateTimeFirstDeliveryMoment) < 13
          and convert(date, dateTimeFirstDeliveryMoment) > promisedDeliveryDate
          and (onTimeDelivery = 1 or onTimeDelivery is null)
      )
      or  -- Should be unknown delivery
      ((datediff(day, orderDate, dateTimeFirstDeliveryMoment) >= 13 or dateTimeFirstDeliveryMoment is null)
       and (onTimeDelivery = 1 or onTimeDelivery = 0)
      );

-- Returns: ALL GOOD
select *
from dbo.data_cleaned
where (datediff(day, orderDate, returnDate) < 30 and noReturn = 1)
      or ((datediff(day, orderDate, returnDate) >= 30 or returnDate is null) and noReturn = 0);

-- Cases: ALL GOOD
select *
from dbo.data_cleaned
where (datediff(day, orderDate, startDateCase) < 30 and noCase = 1)
      or ((datediff(day, orderDate, startDateCase) >= 30 or startDateCase is null) and noCase = 0);

-- Overview
select sum(   case when cancellationReasonCode like 'SELLER%'
                        and datediff(day, orderDate, cancellationDate) < 11
                        and noCancellation = 1 then
                       1
                  else
                      0
              end
          ) as [01. unrecognized SELLER cancellation],                                                                                                 -- occurs 0 times
       sum(   case when cancellationReasonCode like 'SELLER%'
                        and (datediff(day, orderDate, cancellationDate) >= 11 or cancellationDate is null)
                        and noCancellation = 0 then
                       1
                  else
                      0
              end
          ) as [02. mistaken SELLER cancellation],                                                                                                     -- occurs 0 times
       sum(   case when cancellationReasonCode like 'CUST%'
                        and datediff(day, orderDate, cancellationDate) < 11
                        and cancellationDate > promisedDeliveryDate
                        and noCancellation = 1 then
                       1
                  else
                      0
              end
          ) as [03. unrecognized CUSTOMER cancellation],                                                                                               -- occurs 0 times
       sum(   case when cancellationReasonCode like 'CUST%'
                        and
                        (
                            datediff(day, orderDate, cancellationDate) >= 11
                            or cancellationDate <= promisedDeliveryDate
                            or cancellationDate is null
                        )
                        and noCancellation = 0 then
                       1
                  else
                      0
              end
          ) as [04. mistaken CUSTOMER cancellation],                                                                                                   -- occurs 0 times
       sum(   case when cancellationReasonCode not like 'CUST%'
                        and cancellationReasonCode not like 'SELLER%'
                        and datediff(day, orderDate, cancellationDate) < 11
                        and noCancellation = 1 then
                       1
                  else
                      0
              end
          ) as [5. unrecognized OTHER cancellation],                                                                                                   -- occurs 277 times
       sum(   case when cancellationReasonCode not like 'CUST%'
                        and cancellationReasonCode not like 'SELLER%'
                        and (datediff(day, orderDate, cancellationDate) >= 11 or cancellationDate is null)
                        and noCancellation = 0 then
                       1
                  else
                      0
              end
          ) as [06. mistaken OTHER cancellation],                                                                                                      -- occurs 0 times
       sum(   case when datediff(day, orderDate, dateTimeFirstDeliveryMoment) < 13
                        and convert(date, dateTimeFirstDeliveryMoment) <= promisedDeliveryDate
                        and (onTimeDelivery = 0 or onTimeDelivery is null) then
                       1
                  else
                      0
              end
          ) as [07. unrecognized on time delivery],                                                                                                    -- occurs 0 times
       sum(   case when datediff(day, orderDate, dateTimeFirstDeliveryMoment) < 13
                        and convert(date, dateTimeFirstDeliveryMoment) > promisedDeliveryDate
                        and (onTimeDelivery = 1 or onTimeDelivery is null) then
                       1
                  else
                      0
              end
          ) as [08. unrecognized late delivery],                                                                                                       -- occurs 0 times
       sum(   case when (datediff(day, orderDate, dateTimeFirstDeliveryMoment) >= 13 or dateTimeFirstDeliveryMoment is null)
                        and onTimeDelivery in ( 0, 1 ) then
                       1
                  else
                      0
              end
          ) as [09. unrecognized unknown delivery],                                                                                                    -- occurs 0 times
       sum(case when datediff(day, orderDate, returnDate) < 30 and noReturn = 1 then 1 else 0 end) as [10. unrecognized return],                       -- occurs 0 times
       sum(case when (datediff(day, orderDate, returnDate) >= 30 or returnDate is null) and noReturn = 0 then 1 else 0 end) as [11. mistaken return],  -- occurs 0 times
       sum(case when datediff(day, orderDate, startDateCase) < 30 and noCase = 1 then 1 else 0 end) as [12. unrecognized case],                        -- occurs 0 times
       sum(case when (datediff(day, orderDate, startDateCase) >= 30 or startDateCase is null) and noCase = 0 then 1 else 0 end) as [13. mistaken case] -- occurs 0 times
from dbo.data_cleaned;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- CHECK CONSTRUCTION OF MATCH LABELS: ALL FINE (unknown should be further down the tree)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select case when noCancellation = 1
                 and onTimeDelivery = 1
                 and noReturn = 1
                 and noCase = 1 then
                'Happy'
           when noCancellation = 1
                and onTimeDelivery is null
                and noReturn = 1
                and noCase = 1 then
               'Unknown'
           else
               'Unhappy'
       end [Match Labels as computed by Match Determinants],
       generalMatchClassification,
       count(*) [count],
       round(count(*) * 1e2 / (select count(*) from dbo.data_cleaned), 2) [percentage]
from dbo.data_cleaned
group by case when noCancellation = 1
                   and onTimeDelivery = 1
                   and noReturn = 1
                   and noCase = 1 then
                  'Happy'
             when noCancellation = 1
                  and onTimeDelivery is null
                  and noReturn = 1
                  and noCase = 1 then
                 'Unknown'
             else
                 'Unhappy'
         end,
         generalMatchClassification
order by 4 desc;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Additional strange observations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
select *
from dbo.data_cleaned
where quantityReturned > quantityOrdered;

select quantityOrdered,
       quantityReturned,
       quantityReturned - quantityOrdered as [diff],
       count(*) [count]
from dbo.data_cleaned
where quantityReturned > quantityOrdered
group by quantityReturned,
         quantityOrdered,
         quantityReturned - quantityOrdered
order by 3,1;