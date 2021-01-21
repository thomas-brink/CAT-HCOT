use BOL;

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- DATA CLEANING (INCL. COUNTS): 6,516 ORDERS ARE THROWN OUT
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Count is 4,779,466
select count(*) [count] 
from dbo.data_combined

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
	   sum(case when promisedDeliveryDate is null												then 1 else 0 end) as [15. Promised delivery date unknown],				-- occurs  939 times
       count(*) [count]
from dbo.data_combined

-- Delete all noise (non-sensical rows). Total = 6,516 orders (0.14% of total orders)
delete from dbo.data_combined
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
from dbo.data_combined;