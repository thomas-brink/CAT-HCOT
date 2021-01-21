use BOL;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- CHECK CONSTRUCTION OF MATCH DETERMINANTS: MANY UNRECOGNIZED CANCELLATIONS
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Cancellations
select orderDate,
       cancellationDate,
	   promisedDeliveryDate,
       datediff(day, orderDate, cancellationDate) [cancellation time],
	   datediff(day, promisedDeliveryDate, cancellationDate) [time between promised delivery and cancellation],
	   cancellationReasonCode,
       noCancellation,
       *
from dbo.data_combined
where (
          cancellationReasonCode like 'SELLER%'
          and
          ((datediff(day, orderDate, cancellationDate) < 11 and noCancellation = 1)
           or
           ((datediff(day, orderDate, cancellationDate) >= 11 or cancellationDate is null)
            and noCancellation = 0
           )
          )
      )
	  or
	  (
          (cancellationReasonCode not like 'CUST%' and cancellationReasonCode not like 'SELLER%')
          and
          ((datediff(day, orderDate, cancellationDate) < 11 and noCancellation = 1)
           or
           ((datediff(day, orderDate, cancellationDate) >= 11 or cancellationDate is null)
            and noCancellation = 0
           )
          )
      )
      or
      (
          cancellationReasonCode like 'CUST%'
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
      )
order by 1,
         2;

		---- Empirical distribution of unrecognized cancellations
		--select datediff(day, orderDate, cancellationDate) [cancellation time],
		--		case when datediff(day, orderDate, cancellationDate) < 11 then 1 else 0 end [cancellation],
		--		noCancellation,
		--		count(*) [count]
		--from dbo.data_combined
		--where (datediff(day, orderDate, cancellationDate) < 11 and noCancellation = 1)
		--		or ((datediff(day, orderDate, cancellationDate) >= 11 or cancellationDate is null) and noCancellation = 0)
		--group by datediff(day, orderDate, cancellationDate),
		--			case when datediff(day, orderDate, cancellationDate) < 11 then 1 else 0 end,
		--			noCancellation
		--order by 1;

		---- Ratio between correct cancellations and unrecognized cancellations
		--select case when noCancellation = 0 then
		--				'Correct cancellation'
		--			when datediff(day, orderDate, cancellationDate) < 11
		--				and noCancellation = 1 then
		--				'Unrecognized cancellation'
		--			else
		--				'Normal'
		--		end [Cancellation Type],
		--		count(*) [count]
		--from dbo.data_combined
		--where datediff(day, orderDate, cancellationDate) < 11
		--group by case when noCancellation = 0 then
		--					'Correct cancellation'
		--				when datediff(day, orderDate, cancellationDate) < 11
		--					and noCancellation = 1 then
		--					'Unrecognized cancellation'
		--				else
		--					'Normal'
		--			end
		--order by 1;

-- Deliveries
select *
from dbo.data_combined
where (
          datediff(day, orderDate, dateTimeFirstDeliveryMoment) < 13
          and convert(date, dateTimeFirstDeliveryMoment) <= promisedDeliveryDate
          and (onTimeDelivery = 0 or onTimeDelivery is null)
      )
      or
      (
          datediff(day, orderDate, dateTimeFirstDeliveryMoment) < 13
          and convert(date, dateTimeFirstDeliveryMoment) > promisedDeliveryDate
          and (onTimeDelivery = 1 or onTimeDelivery is null)
      )
      or
      ((datediff(day, orderDate, dateTimeFirstDeliveryMoment) >= 13 or dateTimeFirstDeliveryMoment is null)
       and (onTimeDelivery = 1 or onTimeDelivery = 0)
      );

-- Returns
select *
from dbo.data_combined
where (datediff(day, orderDate, returnDate) < 30 and noReturn = 1)
      or ((datediff(day, orderDate, returnDate) >= 30 or returnDate is null) and noReturn = 0);

-- Cases
select *
from dbo.data_combined
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
from dbo.data_combined;

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
       round(count(*) * 1e2 / (select count(*) from dbo.data_combined), 2) [percentage]
from dbo.data_combined
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
from dbo.data_combined
where quantityReturned > quantityOrdered;