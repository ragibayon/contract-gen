import icontract

@icontract.require(lambda s: len(s) > 0)
@icontract.require(lambda s: all(c in '01' for c in s))
@icontract.require(lambda s: len(s) <= 100)
@icontract.require(lambda s: '0' in s and '1' in s)
@icontract.ensure(lambda result: result in ('YES', 'NO'))
@icontract.ensure(lambda s, result: (result == 'YES') == ('0000000' in s or '1111111' in s))
