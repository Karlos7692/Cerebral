

function [RawIn, RawOut, CVIn, CVOut, DroppedRawIn, DroppedRawOut] = rollover(RawIn, RawOut, CVIn, CVOut, DroppedRawIn, DroppedRawOut)
     DroppedRawIn = [DroppedRawIn ; RawIn(1,:)];
     DroppedRawOut = [DroppedRawOut ; RawOut(1,:)];
     RawIn = [RawIn(2:end, :) ; CVIn(1,:)];
     RawOut = [RawOut(2:end, :) ; CVOut(1,:)];
     CVIn = CVIn(2:end,:);
     CVOut = CVOut(2:end,:);
end